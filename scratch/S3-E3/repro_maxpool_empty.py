"""max_pool1d/2d on an input with a zero-size spatial dim aborts on TPU."""
import sys
import torch
import torch.nn.functional as F

shape = tuple(int(s) for s in sys.argv[1].split(",")) if len(sys.argv) > 1 else (1, 10, 0)
for dev in ["cpu", "tpu"]:
  x = torch.randn(shape, device=dev)
  try:
    fn = F.max_pool1d if len(shape) <= 3 else F.max_pool2d
    print(dev, fn(x, kernel_size=1).cpu().shape, flush=True)
  except Exception as e:
    print(dev, type(e).__name__, str(e)[:300], flush=True)
