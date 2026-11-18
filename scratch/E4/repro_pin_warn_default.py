import torch
import torch_tpu  # noqa: F401
import warnings
warnings.simplefilter("error", DeprecationWarning)
torch.ones(4).to("tpu", non_blocking=True)
print("no warning")
