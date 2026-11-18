import torch
import torch_tpu  # noqa: F401
x = torch.randn(4, 4)
try:
    print(torch.compile(lambda t: t.sin() * 2)(x.to("tpu")).cpu() - x.sin() * 2)
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:600])
