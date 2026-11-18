import torch
x = torch.tensor([3, 1, 3, 2, 1])
print("cpu:", torch.unique(x, sorted=False))
print("tpu:", torch.unique(x.to("tpu"), sorted=False).cpu())
