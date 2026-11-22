import torch
import torch_tpu

print("torch_tpu file:", torch_tpu.__file__)
print(torch.get_autocast_dtype("tpu"))
x = torch.full((4, 4), 300.0, device="tpu")
with torch.autocast("tpu"):
  y = x @ x
print(y.dtype, y[0, 0].item())
