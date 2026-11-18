"""Compare autograd grads TPU vs CPU on op_db samples (f32). Args: ops outfile."""
import json
import sys

import torch
import torch_tpu  # noqa: F401
from torch.testing._internal.common_methods_invocations import op_db

torch.set_float32_matmul_precision("highest")
names = sys.argv[1].split(",")
out = sys.argv[2]
db = {}
for o in op_db:
  if o.name in names and o.supports_autograd:
    db.setdefault(o.name, []).append(o)


def flat(x):
  if isinstance(x, torch.Tensor):
    return [x]
  if isinstance(x, (list, tuple)):
    r = []
    for y in x:
      r += flat(y)
    return r
  return []


def rec(d):
  with open(out, "a") as f:
    f.write(json.dumps(d) + "\n")


for name in names:
  for op in db.get(name, []):
    tag = f"{name} {op.variant_test_name}"
    try:
      samples = list(op.sample_inputs("cpu", torch.float32, requires_grad=True))[:12]
    except Exception as e:
      rec({"op": tag, "kind": "sample_err", "msg": repr(e)[:200]})
      continue
    for i, s in enumerate(samples):
      def leaves(smp):
        return [t for t in flat([smp.input, list(smp.args), list(smp.kwargs.values())]) if t.requires_grad]
      try:
        o = [t for t in flat(op(s.input, *s.args, **s.kwargs)) if t.is_floating_point() and t.requires_grad]
        if not o:
          continue
        torch.manual_seed(0)
        cot = [torch.randn_like(t) for t in o]
        ls = leaves(s)
        g_cpu = torch.autograd.grad(o, ls, cot, allow_unused=True)
      except Exception:
        continue
      try:
        st = s.transform(lambda t: t.detach().to("tpu").requires_grad_(t.requires_grad) if isinstance(t, torch.Tensor) else t)
        ot = [t for t in flat(op(st.input, *st.args, **st.kwargs)) if t.is_floating_point() and t.requires_grad]
        lt = leaves(st)
        g_tpu = torch.autograd.grad(ot, lt, [c.to("tpu") for c in cot], allow_unused=True)
      except Exception as e:
        rec({"op": tag, "i": i, "kind": "raise", "msg": repr(e).split("\n")[0][:300]})
        continue
      for j, (a, b) in enumerate(zip(g_cpu, g_tpu)):
        if a is None or b is None:
          if (a is None) != (b is None):
            rec({"op": tag, "i": i, "kind": "none_mismatch"})
          continue
        try:
          torch.testing.assert_close(b.detach().cpu(), a, atol=1e-4, rtol=1e-4, equal_nan=True)
        except AssertionError as e:
          desc = str([tuple(t.shape) if isinstance(t, torch.Tensor) else t for t in [s.input] + list(s.args)])[:200]
          rec({"op": tag, "i": i, "g": j, "kind": "value", "msg": " | ".join(str(e).split("\n")[:4])[:300], "desc": desc, "kw": str(s.kwargs)[:200]})
rec({"op": "__END__"})
