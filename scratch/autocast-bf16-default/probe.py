import threading
import torch, torch_tpu  # noqa: F401
print("torch_tpu file:", torch_tpu.__file__)
print("main default:", torch.get_autocast_dtype("tpu"))
a = torch.randn(4, 4, device="tpu", requires_grad=True)
with torch.autocast("tpu"):
  y = a @ a
print("main mm under autocast:", y.dtype)
x = torch.full((4, 4), 300.0, device="tpu")
with torch.autocast("tpu"):
  print("300x300 sum of 4:", (x @ x).float().cpu()[0, 0].item())
seen = {}
class Rec(torch.autograd.Function):
  @staticmethod
  def forward(ctx, t): return t.clone()
  @staticmethod
  def backward(ctx, g):
    seen["bw_thread"] = threading.current_thread().name
    seen["bw_dtype"] = torch.get_autocast_dtype("tpu")
    seen["bw_enabled"] = torch.is_autocast_enabled("tpu")
    return g
Rec.apply(a).sum().backward()
print("backward outside autocast:", seen)
seen.clear()
with torch.autocast("tpu"):
  out = Rec.apply(a).sum()
out.backward()
print("backward of autocast fwd:", seen)
seen.clear()
with torch.autocast("tpu"):
  Rec.apply(a).sum().backward()
print("backward inside autocast:", seen)
res = {}
def w():
  res["dtype"] = torch.get_autocast_dtype("tpu")
  b = torch.randn(4, 4, device="tpu")
  with torch.autocast("tpu"):
    res["mm"] = (b @ b).dtype
t = threading.Thread(target=w); t.start(); t.join()
print("new thread:", res)
@torch.compile
def f(p, q):
  with torch.autocast("tpu"):
    return p @ q
try:
  print("compile mm under autocast:", f(a.detach(), a.detach()).dtype)
except Exception as e:
  print("compile error:", type(e).__name__, str(e)[:300])
# GradScaler
s = torch.amp.GradScaler("tpu")
print("GradScaler enabled:", s.is_enabled())
