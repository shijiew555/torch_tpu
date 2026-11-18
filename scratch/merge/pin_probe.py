import warnings
import torch, torch_tpu  # noqa: F401
p = torch.ones(4).pin_memory()
print("is_pinned:", p.is_pinned())
warnings.simplefilter("error")
y = p.to("tpu", non_blocking=True)
print("pinned non_blocking copy OK, no warning:", y.cpu().tolist())
