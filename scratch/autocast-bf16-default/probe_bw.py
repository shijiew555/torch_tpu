import threading
import torch, torch_tpu  # noqa: F401
a = torch.randn(4, 4, device="tpu", requires_grad=True)
def g(p, q):
  with torch.autocast("tpu"):
    return p @ q
def show(tag):
  print(tag, torch.get_autocast_dtype("tpu"))
show("start")
print("compile1", torch.compile(g)(a.detach(), a.detach()).dtype)
show("after compile1")
(a * 2).sum().backward()
show("after plain backward")
torch._dynamo.reset()
print("compile2", torch.compile(g)(a.detach(), a.detach()).dtype)
with torch.autocast("tpu"):
  y = a @ a
show("after eager autocast")
torch._dynamo.reset()
print("compile3", torch.compile(g)(a.detach(), a.detach()).dtype)
torch._dynamo.reset()
print("compile4 grad", torch.compile(g)(a, a).dtype)
