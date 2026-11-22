import faulthandler, os, tempfile, traceback
import torch, torch_tpu  # noqa: F401
import torch.distributed as dist
from torch import nn
faulthandler.dump_traceback_later(600, exit=True)
torch.set_float32_matmul_precision("highest")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29544")
dist.init_process_group("tpu", rank=0, world_size=1)
T = "tpu"
def run(name, f):
  try: print(f"== {name}: {f()}", flush=True)
  except Exception as e:
    print(f"== {name}: FAIL {type(e).__name__}: {str(e)[:300]}", flush=True); traceback.print_exc(limit=-3)
def mk(dev):
  torch.manual_seed(0)
  return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2), nn.Linear(2, 2)).to(dev)
def train(m, dev, unused=False):
  opt = torch.optim.SGD(m.parameters(), lr=0.1); torch.manual_seed(1)
  for _ in range(3):
    x = torch.randn(4, 8).to(dev); opt.zero_grad()
    mod = m.module if hasattr(m, "module") else m
    out = m(x) if not unused else mod[2](mod[1](mod[0](x)))
    out.pow(2).sum().backward(); opt.step()
  return [p.detach().full_tensor() if hasattr(p, "full_tensor") else p.detach() for p in m.parameters()]
def d(a, b): return max((x.float().cpu() - y).abs().max().item() for x, y in zip(a, b))
from torch.nn.parallel import DistributedDataParallel as DDP
run("ddp find_unused", lambda: d(train(DDP(mk(T), find_unused_parameters=True), T, True), train(mk("cpu"), "cpu", True)))
run("ddp bucket_view", lambda: d(train(DDP(mk(T), gradient_as_bucket_view=True), T), train(mk("cpu"), "cpu")))
run("ddp static_graph", lambda: d(train(DDP(mk(T), static_graph=True), T), train(mk("cpu"), "cpu")))
def ddp_hook():
  from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as dh
  m = DDP(mk(T)); m.register_comm_hook(None, dh.bf16_compress_hook)
  return d(train(m, T), train(mk("cpu"), "cpu"))
run("ddp bf16 hook", ddp_hook)
def fsdp_ckpt():
  from torch.distributed.device_mesh import init_device_mesh
  from torch.distributed.fsdp import fully_shard
  from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
  mesh = init_device_mesh(T, (1,))
  m = mk(T); m[0] = checkpoint_wrapper(m[0]); fully_shard(m[0], mesh=mesh); fully_shard(m, mesh=mesh)
  return d(train(m, T), train(mk("cpu"), "cpu"))
run("fsdp2 + checkpoint_wrapper", fsdp_ckpt)
def fsdp_noreshard():
  from torch.distributed.device_mesh import init_device_mesh
  from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy
  mesh = init_device_mesh(T, (1,))
  m = mk(T); fully_shard(m, mesh=mesh, reshard_after_forward=False)
  return d(train(m, T), train(mk("cpu"), "cpu"))
run("fsdp2 reshard_after_forward=False", fsdp_noreshard)
def fsdp_offload():
  from torch.distributed.device_mesh import init_device_mesh
  from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy
  mesh = init_device_mesh(T, (1,))
  m = mk(T); fully_shard(m, mesh=mesh, offload_policy=CPUOffloadPolicy(pin_memory=False))
  return d(train(m, "tpu"), train(mk("cpu"), "cpu"))
run("fsdp2 cpu offload", fsdp_offload)
def dcp_async():
  import torch.distributed.checkpoint as dcp
  m = mk(T); dd = tempfile.mkdtemp()
  fut = dcp.async_save(m.state_dict(), checkpoint_id=dd); fut.result()
  sd = mk(T).state_dict(); [v.zero_() for v in sd.values()]
  dcp.load(sd, checkpoint_id=dd)
  return max((a - b).abs().max().item() for a, b in zip(m.state_dict().values(), sd.values()))
run("dcp async_save", dcp_async)
def clip_dtensor():
  from torch.distributed.device_mesh import init_device_mesh
  from torch.distributed.fsdp import fully_shard
  mesh = init_device_mesh(T, (1,))
  m = mk(T); fully_shard(m, mesh=mesh)
  m(torch.ones(2, 8, device=T)).sum().backward()
  n = nn.utils.clip_grad_norm_(m.parameters(), 0.1)
  mc = mk("cpu"); mc(torch.ones(2, 8)).sum().backward()
  return type(n).__name__, n.full_tensor().item() if hasattr(n, "full_tensor") else n.item(), nn.utils.clip_grad_norm_(mc.parameters(), 0.1).item()
run("clip_grad_norm_ fsdp2", clip_dtensor)
dist.destroy_process_group(); print("done")
