import torch
import torch_tpu  # noqa: F401

x = torch.tensor([3, 1, 3, 2, 1])
print("cpu sorted=False:", torch.unique(x, sorted=False))
try:
  print("tpu sorted=False:", torch.unique(x.to("tpu"), sorted=False).cpu())
  v, inv, cnt = torch.unique(x.to("tpu"), sorted=False, return_inverse=True,
                             return_counts=True)
  print("tpu sorted=False inverse/counts:", v.cpu(), inv.cpu(), cnt.cpu())
except RuntimeError as e:
  print("tpu sorted=False: RuntimeError:", str(e).splitlines()[0])
