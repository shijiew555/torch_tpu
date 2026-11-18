import torch
import torch_tpu  # noqa: F401

batch = torch.ones(4)  # not pinned, as from a DataLoader with pin_memory=False
y = batch.to("tpu", non_blocking=True)
print(y.cpu())
