import torch
import torch_tpu  # noqa: F401

torch.manual_seed(0)
fails = 0
for dtype in [torch.float32, torch.bfloat16, torch.int64, torch.bool]:
  for ri in [True, False]:
    for rc in [True, False]:
      if dtype.is_floating_point:
        x = torch.randn((2, 10, 5), dtype=dtype)
      elif dtype == torch.bool:
        x = torch.randint(0, 2, (2, 10, 5)).to(torch.bool)
      else:
        x = torch.randint(0, 10, (2, 10, 5), dtype=dtype)
      ref = torch.unique(x, sorted=True, return_inverse=ri, return_counts=rc)
      try:
        out = torch.unique(x.to("tpu"), sorted=False, return_inverse=ri,
                           return_counts=rc)
        out = [o.cpu() for o in out] if isinstance(out, tuple) else [out.cpu()]
        ref = list(ref) if isinstance(ref, tuple) else [ref]
        for o, r in zip(out, ref):
          torch.testing.assert_close(o, r)
        print("OK  ", dtype, ri, rc)
      except Exception as e:
        fails += 1
        print("FAIL", dtype, ri, rc, type(e).__name__, str(e).splitlines()[0])
print("failures:", fails, "of 16")
