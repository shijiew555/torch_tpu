import faulthandler; faulthandler.dump_traceback_later(300, exit=True)
import torch, torch_tpu  # noqa
import torch.nn as nn
acc = torch.accelerator; d="tpu"
def s(tag): acc.synchronize(); print(f"{tag:28s} alloc={acc.memory_allocated()/2**20:9.2f}MiB peak={acc.max_memory_allocated()/2**20:9.2f}MiB reserved={acc.memory_reserved()/2**20:9.2f}MiB")
s("start")
a = torch.empty(1024,1024,64, device=d); s("empty 256MiB")
a.fill_(1); s("after fill_")
b = torch.ones(1024,1024,64, device=d); s("ones 256MiB")
c = a + b; s("a+b")
print("c sum", c[0,0,:2].cpu())
del a, b, c; s("del all")
acc.reset_peak_memory_stats(); s("reset peak")
m = nn.Linear(4096,4096).to(d); s("linear 64MiB")
x = torch.randn(512,4096, device=d)
m(x).sum().backward(); s("after fwd/bwd")
st = acc.memory_stats(); print({k:v for k,v in st.items() if k.startswith("allocated_bytes.all") or k.startswith("num_alloc")})
