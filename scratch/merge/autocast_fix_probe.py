import threading
import torch, torch_tpu  # noqa: F401
torch.set_autocast_dtype("tpu", torch.bfloat16)
a = torch.randn(4, 4, device="tpu", requires_grad=True)
with torch.autocast("tpu"):
    y = a @ a
print("main thread mm dtype:", y.dtype)
res = {}
def worker():
    res["dtype"] = torch.get_autocast_dtype("tpu")
t = threading.Thread(target=worker); t.start(); t.join()
print("new thread default:", res["dtype"])
