import threading, traceback
import torch, torch_tpu  # noqa: F401
import torch.amp.autocast_mode as am
orig = am._enter_autocast
def spy(*args):
  print("enter_autocast thread=%s args=%s tls=%s" % (threading.current_thread().name, args, torch.get_autocast_dtype("tpu")))
  if torch.get_autocast_dtype("tpu") == torch.float16:
    traceback.print_stack(limit=30)
  return orig(*args)
am._enter_autocast = spy
torch.amp._enter_autocast = spy
a = torch.randn(4, 4, device="tpu", requires_grad=True)
def g(p, q):
  with torch.autocast("tpu"):
    return p @ q
print("compile1", torch.compile(g)(a.detach(), a.detach()).dtype)
