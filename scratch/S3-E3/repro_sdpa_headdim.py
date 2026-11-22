"""SDPA with mismatched q/k head dims aborts the process on TPU."""
import torch
import torch.nn.functional as F

for dev in ["cpu", "tpu"]:
  q = torch.ones(1, 2, 4, 8, device=dev)
  k = torch.ones(1, 2, 4, 6, device=dev)
  v = torch.ones(1, 2, 4, 8, device=dev)
  try:
    print(dev, F.scaled_dot_product_attention(q, k, v).cpu().shape, flush=True)
  except Exception as e:
    print(dev, type(e).__name__, str(e)[:300], flush=True)
