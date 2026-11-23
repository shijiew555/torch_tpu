import torch
import torch_tpu  # noqa: F401

print("torch_tpu from", torch_tpu.__file__)

def f(x):
    y = x * 2
    return y, y.view(2, 8), y[1:]

x = torch.arange(16.0).view(4, 4).to("tpu")
a, b, c = torch.compile(f, backend="tpu")(x)
a.add_(100)
print("b[0,:3] =", b[0, :3].tolist(), "(eager expects [100.0, 102.0, 104.0])")
print("shared storage:", a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr() == c.untyped_storage().data_ptr())
