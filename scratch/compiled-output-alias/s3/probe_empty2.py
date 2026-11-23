import torch
import torch_tpu  # noqa: F401

cases = {
    "y,y[1:],y[:0]": lambda x: ((y := x * 2), y[1:], y[:0]),
    "zeros0,y,y[1:]": lambda x: (x.new_zeros(0), (y := x * 2), y[1:]),
    "y,y[1:],zeros0": lambda x: ((y := x * 2), y[1:], x.new_zeros(0)),
}
for name, f in cases.items():
  torch.compiler.reset()
  x = torch.arange(16.0).view(4, 4).to("tpu")
  outs = torch.compile(f, backend="tpu")(x)
  yi = 0 if name.startswith("y") else 1
  outs[yi].add_(100)
  print(name, outs[yi + 1][0, :2].tolist(), [o.untyped_storage().data_ptr() for o in outs])
