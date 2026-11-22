"""torch.utils.checkpoint does not replay TPU dropout masks in backward."""
import sys
import torch
import torch_tpu  # noqa: F401
from torch import nn
import torch.utils.checkpoint as cp

print("hasattr(torch.tpu, '_initialized'):", hasattr(torch.tpu, "_initialized"))
if len(sys.argv) > 1:  # Simulated fix.
  torch.tpu._initialized = True

for dev in ("cpu", "tpu"):
  for reentrant in (True, False):
    torch.manual_seed(0)
    lin = nn.Linear(8, 8).to(dev)
    drop = nn.Dropout(0.5)
    f = lambda x: drop(lin(x))
    x = torch.randn(4, 8, device=dev, requires_grad=True)
    torch.manual_seed(1)
    cp.checkpoint(f, x, use_reentrant=reentrant).sum().backward()
    g_ckpt = x.grad.clone()
    x.grad = None
    torch.manual_seed(1)
    f(x).sum().backward()
    print(f"{dev} use_reentrant={reentrant}: grad diff vs no checkpoint =",
          (g_ckpt - x.grad).abs().max().item())
