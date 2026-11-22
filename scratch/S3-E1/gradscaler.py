"""torch.amp.GradScaler fails on TPU: the two AMP kernels are missing."""
import torch
import torch_tpu  # noqa: F401
from torch import nn

m = nn.Linear(8, 2).to("tpu")
opt = torch.optim.SGD(m.parameters(), lr=0.1)
scaler = torch.amp.GradScaler("tpu")
with torch.autocast("tpu", dtype=torch.float16):
  loss = m(torch.ones(4, 8, device="tpu")).sum()
scaler.scale(loss).backward()
scaler.step(opt)  # Raises here.
scaler.update()
print("scale", scaler.get_scale())
