import threading
import torch, torch_tpu  # noqa: F401
print("main default:", torch.get_autocast_dtype("tpu"))
a = torch.randn(4, 4, device="tpu")
def g(p, q):
  with torch.autocast("tpu"):
    return p @ q
for be in ["eager", "aot_eager", None]:
  torch._dynamo.reset()
  f = torch.compile(g) if be is None else torch.compile(g, backend=be)
  try:
    print("backend", be, "->", f(a, a).dtype)
  except Exception as e:
    print("backend", be, "error", type(e).__name__, str(e)[:200])
torch._dynamo.reset()
seen = []
import torch.fx
def be_spy(gm, ex):
  seen.append((threading.current_thread().name, torch.get_autocast_dtype("tpu")))
  print(gm.code)
  return gm
print("spy ->", torch.compile(g, backend=be_spy)(a, a).dtype, seen)
