"""Single-chip distributed checks: PG, DDP, FSDP2, DTensor, DCP, funcol."""
import faulthandler
import os
import sys
import tempfile
import traceback

import torch
import torch_tpu  # noqa: F401
import torch.distributed as dist
from torch import nn

faulthandler.dump_traceback_later(600, exit=True)
torch.set_float32_matmul_precision("highest")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29533")
backend = sys.argv[1] if len(sys.argv) > 1 else "tpu"
dist.init_process_group(backend=backend, rank=0, world_size=1)
print("backend", dist.get_backend())
T = "tpu"


def step(name, f):
  try:
    print(f"== {name}: {f()}", flush=True)
  except Exception as e:  # noqa: BLE001
    print(f"== {name}: FAIL {type(e).__name__}: {str(e)[:400]}", flush=True)
    traceback.print_exc(limit=-4)


def mk(dev):
  torch.manual_seed(0)
  return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(dev)


def train(m, dev, n=3):
  opt = torch.optim.Adam(m.parameters(), lr=0.01)
  torch.manual_seed(1)
  for _ in range(n):
    x = torch.randn(4, 8).to(dev)
    opt.zero_grad()
    m(x).pow(2).sum().backward()
    opt.step()
  return [p.detach().full_tensor() if hasattr(p, "full_tensor") else p.detach()
          for p in m.parameters()]


REF = train(mk("cpu"), "cpu")


def diff(ps):
  return max((a.float().cpu() - b).abs().max().item() for a, b in zip(ps, REF))


def collectives():
  t = torch.arange(4.0, device=T)
  out = []
  dist.all_reduce(t)
  out.append(t.cpu().tolist())
  g = [torch.empty(4, device=T)]
  dist.all_gather(g, t)
  out.append(g[0].cpu().tolist())
  o = torch.empty(4, device=T)
  dist.all_gather_into_tensor(o, t)
  dist.reduce_scatter_tensor(o, t)
  dist.broadcast(t, 0)
  out.append(o.cpu().tolist())
  w = dist.all_reduce(t, async_op=True)
  w.wait()
  objs = [None]
  dist.all_gather_object(objs, {"a": 1})
  out.append(objs)
  ol = [{"k": 2}]
  dist.broadcast_object_list(ol, src=0)
  out.append(ol)
  dist.barrier()
  return out


def funcol():
  import torch.distributed._functional_collectives as fc
  t = torch.arange(4.0, device=T)
  a = fc.all_reduce(t, "sum", dist.group.WORLD)
  b = fc.all_gather_tensor(t, 0, dist.group.WORLD)
  c = fc.reduce_scatter_tensor(t, "sum", 0, dist.group.WORLD)
  return [x.cpu().tolist() for x in (a, b, c)]


def ddp():
  from torch.nn.parallel import DistributedDataParallel as DDP
  m = DDP(mk(T))
  return diff(train(m, T))


def ddp_no_sync_accum():
  from torch.nn.parallel import DistributedDataParallel as DDP
  m = DDP(mk(T))
  with m.no_sync():
    m(torch.ones(2, 8, device=T)).sum().backward()
  m(torch.ones(2, 8, device=T)).sum().backward()
  return m.module[0].weight.grad.abs().sum().item()


def fsdp2():
  from torch.distributed.device_mesh import init_device_mesh
  from torch.distributed.fsdp import fully_shard
  mesh = init_device_mesh(T, (1,))
  m = mk(T)
  for layer in m:
    if isinstance(layer, nn.Linear):
      fully_shard(layer, mesh=mesh)
  fully_shard(m, mesh=mesh)
  return diff(train(m, T))


def fsdp2_mixed():
  from torch.distributed.device_mesh import init_device_mesh
  from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
  mesh = init_device_mesh(T, (1,))
  m = mk(T)
  mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                            reduce_dtype=torch.float32)
  fully_shard(m, mesh=mesh, mp_policy=mp)
  return diff(train(m, T))


def dtensor():
  from torch.distributed.device_mesh import init_device_mesh
  from torch.distributed.tensor import Replicate, Shard, distribute_tensor
  mesh = init_device_mesh(T, (1,))
  x = torch.randn(4, 6)
  d = distribute_tensor(x.to(T), mesh, [Shard(0)])
  y = (d @ d.T).relu().sum(1)
  r = d.redistribute(mesh, [Replicate()])
  return ((y.full_tensor().cpu() - (x @ x.T).relu().sum(1)).abs().max().item(),
          type(y).__name__, r.placements)


def tp():
  from torch.distributed.device_mesh import init_device_mesh
  from torch.distributed.tensor.parallel import (ColwiseParallel,
                                                 RowwiseParallel,
                                                 parallelize_module)
  mesh = init_device_mesh(T, (1,))
  m = mk(T)
  parallelize_module(m, mesh, {"0": ColwiseParallel(), "2": RowwiseParallel()})
  return diff(train(m, T))


def dcp():
  import torch.distributed.checkpoint as dcp_mod
  from torch.distributed.checkpoint.state_dict import (get_state_dict,
                                                       set_state_dict)
  from torch.distributed.device_mesh import init_device_mesh
  from torch.distributed.fsdp import fully_shard
  mesh = init_device_mesh(T, (1,))
  m = mk(T)
  fully_shard(m, mesh=mesh)
  opt = torch.optim.AdamW(m.parameters(), lr=0.01)
  m(torch.ones(2, 8, device=T)).sum().backward()
  opt.step()
  msd, osd = get_state_dict(m, opt)
  d = tempfile.mkdtemp()
  dcp_mod.save({"m": msd, "o": osd}, checkpoint_id=d)
  m2 = mk(T)
  fully_shard(m2, mesh=mesh)
  opt2 = torch.optim.AdamW(m2.parameters(), lr=0.01)
  m2(torch.ones(2, 8, device=T)).sum().backward()
  opt2.step()
  opt2.zero_grad()
  msd2, osd2 = get_state_dict(m2, opt2)
  sd = {"m": msd2, "o": osd2}
  dcp_mod.load(sd, checkpoint_id=d)
  set_state_dict(m2, opt2, model_state_dict=sd["m"], optim_state_dict=sd["o"])
  a = [p.full_tensor().cpu() for p in m.parameters()]
  b = [p.full_tensor().cpu() for p in m2.parameters()]
  return max((x - y).abs().max().item() for x, y in zip(a, b))


def dcp_plain():
  import torch.distributed.checkpoint as dcp_mod
  m = mk(T)
  d = tempfile.mkdtemp()
  dcp_mod.save(m.state_dict(), checkpoint_id=d)
  m2 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(T)
  sd = m2.state_dict()
  dcp_mod.load(sd, checkpoint_id=d)
  m2.load_state_dict(sd)
  return max((x - y).abs().max().item()
             for x, y in zip(m.parameters(), m2.parameters()))


only = sys.argv[2:]
for f in (collectives, funcol, ddp, ddp_no_sync_accum, fsdp2, fsdp2_mixed,
          dtensor, tp, dcp, dcp_plain):
  if only and f.__name__ not in only:
    continue
  step(f.__name__, f)
dist.destroy_process_group()
print("done")
