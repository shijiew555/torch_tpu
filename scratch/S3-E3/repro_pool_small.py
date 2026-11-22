"""Pooling with a window larger than the (padded) input; one case per process."""
import sys
import torch
import torch.nn.functional as F

CASES = {
  "max1d": lambda d: F.max_pool1d(torch.ones(1, 1, 2, device=d), 3),
  "max2d": lambda d: F.max_pool2d(torch.ones(1, 1, 2, 2, device=d), 3),
  "max2d_grad": lambda d: F.max_pool2d(torch.ones(1, 1, 2, 2, device=d, requires_grad=True), 3),
  "max2d_module": lambda d: torch.nn.MaxPool2d(3)(torch.ones(1, 1, 2, 2, device=d)),
  "max2d_indices": lambda d: F.max_pool2d(torch.ones(1, 1, 2, 2, device=d), 3, return_indices=True),
  "max3d": lambda d: F.max_pool3d(torch.ones(1, 1, 2, 2, 2, device=d), 3),
  "avg2d": lambda d: F.avg_pool2d(torch.ones(1, 1, 2, 2, device=d), 3),
  "avg1d": lambda d: F.avg_pool1d(torch.ones(1, 1, 2, device=d), 3),
  "avg3d": lambda d: F.avg_pool3d(torch.ones(1, 1, 2, 2, 2, device=d), 3),
  "max2d_empty_spatial": lambda d: F.max_pool2d(torch.ones(1, 1, 0, 4, device=d), 1),
  "avg2d_empty_spatial": lambda d: F.avg_pool2d(torch.ones(1, 1, 0, 4, device=d), 1),
}
f = CASES[sys.argv[1]]
for dev in ["cpu", "tpu"]:
  try:
    r = f(dev)
    r = r[0] if isinstance(r, tuple) else r
    print(dev, "ok", tuple(r.cpu().shape), flush=True)
  except Exception as e:
    print(dev, type(e).__name__, str(e).replace("\n", " | ")[:250], flush=True)
