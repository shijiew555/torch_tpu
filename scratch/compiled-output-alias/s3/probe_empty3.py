import torch
import torch_tpu  # noqa: F401
from torch_tpu._internal.export import export as ex

orig = ex._find_output_views
def spy(flat):
  for o in flat:
    if isinstance(o, torch.Tensor):
      print("  out", tuple(o.shape), o.stride(), o.storage_offset(), "key", ex._storage_key(o), "nbytes", o.untyped_storage().nbytes(), "covers", ex._covers_storage(o))
  r = orig(flat)
  print("  bases", [(i, tuple(b.shape)) for i, b in r[0]], "views", r[1])
  return r
ex._find_output_views = spy

cases = {
    "z0,z04,y,y1,y0": lambda x: (x.new_zeros(0), x.new_zeros(0, 4), (y := x * 2), y[1:], y[:0]),
    "z0,z04,y,y1": lambda x: (x.new_zeros(0), x.new_zeros(0, 4), (y := x * 2), y[1:]),
}
for name, f in cases.items():
  torch.compiler.reset()
  print(name)
  x = torch.arange(16.0).view(4, 4).to("tpu")
  outs = torch.compile(f, backend="tpu")(x)
  outs[2].add_(100)
  print(name, outs[3][0, :2].tolist())
