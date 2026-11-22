"""Index an empty tensor with an empty index tensor on TPU."""
import faulthandler
import sys

import torch
import torch_tpu  # noqa: F401

faulthandler.dump_traceback_later(120, exit=True)
case = sys.argv[1]
d = "tpu"
if case == "long_empty_idx":
  x = torch.empty(0, dtype=torch.long, device=d)
  i = torch.empty(0, dtype=torch.long, device=d)
  print(x[i].cpu())
elif case == "float_empty_idx":
  x = torch.empty(0, device=d)
  i = torch.empty(0, dtype=torch.long, device=d)
  print(x[i].cpu())
elif case == "empty_2d_rows":
  x = torch.empty(0, 4, device=d)
  i = torch.empty(0, dtype=torch.long, device=d)
  print(x[i].cpu())
elif case == "index_select":
  x = torch.empty(0, device=d)
  i = torch.empty(0, dtype=torch.long, device=d)
  print(torch.index_select(x, 0, i).cpu())
elif case == "nonempty_empty_idx":
  x = torch.arange(5, device=d)
  i = torch.empty(0, dtype=torch.long, device=d)
  print(x[i].cpu())
elif case == "bool_mask_then_index":
  # Pattern from torchvision detection losses: select foreground, then index.
  labels = torch.tensor([1], device=d)
  matched = torch.tensor([-1, -1, -1], device=d)
  fg = matched >= 0
  print(labels[matched[fg]].cpu())
