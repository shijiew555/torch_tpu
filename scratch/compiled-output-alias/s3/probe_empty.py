import torch
import torch_tpu  # noqa: F401


def f(x):
  y = x * 2
  return x.new_zeros(0), x.new_zeros(0, 4), y, y[1:], y[:0]


for dev, backend in (("cpu", None), ("tpu", "tpu")):
  x = torch.arange(16.0).view(4, 4).to(dev)
  fn = torch.compile(f, backend=backend) if backend else f
  outs = fn(x)
  outs[2].add_(100)
  print(dev, [tuple(o.shape) for o in outs], [o.untyped_storage().data_ptr() == outs[0].untyped_storage().data_ptr() for o in outs], outs[3][0, :2].tolist())
