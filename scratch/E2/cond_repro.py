"""torch.cond with a tensor predicate fails under torch.compile(backend="tpu")."""
import torch
import torch_tpu  # noqa: F401

print("torch_tpu:", torch_tpu.__file__)


def f(x):
    return torch.cond(x.sum() > 0, lambda t: t.sin(), lambda t: t.cos(), (x,))


x = torch.rand(4)
print("cpu eager   :", f(x))
print("tpu eager   :", f(x.to("tpu")).cpu())
print("cpu compiled:", torch.compile(f, backend="aot_eager")(x))
print("tpu compiled:", torch.compile(f, backend="tpu")(x.to("tpu")).cpu())
