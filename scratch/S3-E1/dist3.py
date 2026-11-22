import faulthandler, os, tempfile, traceback
import torch, torch_tpu  # noqa: F401
import torch.distributed as dist
from torch import nn
faulthandler.dump_traceback_later(300, exit=True)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29555")
dist.init_process_group("cpu:gloo,tpu:tpu", rank=0, world_size=1)
print("backend", dist.get_backend())
t = torch.ones(3, device="tpu"); dist.all_reduce(t); print("tpu all_reduce", t.cpu())
c = torch.ones(3); dist.all_reduce(c); print("cpu all_reduce", c)
import torch.distributed.checkpoint as dcp
m = nn.Linear(4, 4).to("tpu"); dd = tempfile.mkdtemp()
fut = dcp.async_save(m.state_dict(), checkpoint_id=dd); fut.result()
sd = nn.Linear(4, 4).to("tpu").state_dict()
dcp.load(sd, checkpoint_id=dd)
print("async_save roundtrip diff", max((a - b).abs().max().item() for a, b in zip(m.state_dict().values(), sd.values())))
from torch.nn.parallel import DistributedDataParallel as DDP
mm = DDP(nn.Linear(4, 4).to("tpu")); mm(torch.ones(2, 4, device="tpu")).sum().backward(); print("ddp ok")
dist.destroy_process_group(); print("done")
