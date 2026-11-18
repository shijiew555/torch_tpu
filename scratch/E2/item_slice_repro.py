"""A slice length from .item() on a TPU tensor fails in torch.compile(backend="tpu")."""
import torch
import torch_tpu  # noqa: F401

torch._dynamo.config.capture_scalar_outputs = True
print("torch_tpu:", torch_tpu.__file__)


def f(x, n):
    k = n.item()
    return x[:k].sum()


x = torch.arange(8.0)
n = torch.tensor(5)
print("cpu aot_eager:", torch.compile(f, backend="aot_eager")(x, n))
print("tpu compiled :", torch.compile(f, backend="tpu")(x.to("tpu"), n.to("tpu")).cpu())
