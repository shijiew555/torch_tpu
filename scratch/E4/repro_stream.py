import torch
import torch_tpu  # noqa: F401

print(torch.Stream("tpu:0"))  # works
print(torch.Stream(device=torch.accelerator.current_accelerator()))  # fails
