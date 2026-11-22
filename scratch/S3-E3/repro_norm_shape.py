"""Norm ops with a wrong normalized_shape or weight shape: CPU vs TPU."""
import torch
import torch.nn.functional as F

CASES = {
  "layer_norm_shape": lambda d: F.layer_norm(torch.randn(2, 3, device=d), (4,)),
  "layer_norm_shape_same_numel": lambda d: F.layer_norm(torch.randn(2, 6, device=d), (3, 2)),
  "layer_norm_module": lambda d: torch.nn.LayerNorm(768).to(d)(torch.randn(2, 512, device=d)),
  "layer_norm_weight": lambda d: F.layer_norm(torch.randn(2, 3, device=d), (3,), weight=torch.ones(4, device=d)),
  "rms_norm_shape": lambda d: F.rms_norm(torch.randn(2, 3, device=d), (4,)),
  "rms_norm_weight": lambda d: F.rms_norm(torch.randn(2, 3, device=d), (3,), weight=torch.ones(4, device=d)),
  "group_norm_groups": lambda d: F.group_norm(torch.randn(2, 6, 4, device=d), 4),
  "group_norm_weight": lambda d: F.group_norm(torch.randn(2, 6, 4, device=d), 3, weight=torch.ones(5, device=d)),
}
for n, f in CASES.items():
  out = []
  for dev in ["cpu", "tpu"]:
    try:
      r = f(dev)
      out.append(f"{dev}: ok shape={tuple(r.shape)} sum={r.cpu().sum().item():.4f}")
    except Exception as e:
      out.append(f"{dev}: {type(e).__name__}: " + str(e).replace("\n", " | ")[:200])
  print(n, "\n  " + "\n  ".join(out), flush=True)
