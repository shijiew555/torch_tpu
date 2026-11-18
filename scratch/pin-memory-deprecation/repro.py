import torch
import torch_tpu  # noqa: F401

torch.ones(4).to("tpu", non_blocking=True)
print("no warning")
