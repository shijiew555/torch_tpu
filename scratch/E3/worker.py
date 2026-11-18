"""Sweep op_db samples on TPU vs CPU. Args: opsfile donefile outfile budget_s."""
import json
import sys
import time
import traceback

import torch
import torch_tpu  # noqa: F401
from torch.testing._internal.common_methods_invocations import op_db

torch.set_float32_matmul_precision("highest")

opsfile, donefile, outfile, budget = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
wanted = [l.split() for l in open(opsfile) if l.strip()]  # name variant dtype
try:
  done = set(l.strip() for l in open(donefile))
except FileNotFoundError:
  done = set()
db = {(o.name, o.variant_test_name): o for o in op_db}
start = time.time()
DT = {"f32": torch.float32, "bf16": torch.bfloat16, "i64": torch.int64,
      "f16": torch.float16, "i32": torch.int32, "bool": torch.bool}


def to_tpu(x):
  return x.to("tpu") if isinstance(x, torch.Tensor) else x


def cpuize(x):
  if isinstance(x, torch.Tensor):
    return x.detach().cpu()
  if isinstance(x, (list, tuple)):
    return type(x)(cpuize(y) for y in x) if not hasattr(x, "_fields") else tuple(cpuize(y) for y in x)
  return x


def flat(x):
  if isinstance(x, torch.Tensor):
    return [x]
  if isinstance(x, (list, tuple)):
    r = []
    for y in x:
      r += flat(y)
    return r
  return []


def record(d):
  with open(outfile, "a") as f:
    f.write(json.dumps(d) + "\n")


for key in wanted:
  name, var, dt = key[0], (key[1] if key[1] != "-" else ""), key[2]
  tag = " ".join(key)
  if tag in done:
    continue
  if time.time() - start > budget:
    sys.exit(3)
  with open(donefile, "a") as f:
    f.write(tag + "\n")  # mark first: a crash will skip it next time
  op = db.get((name, var))
  if op is None:
    record({"op": tag, "kind": "missing_in_op_db"})
    continue
  dtype = DT[dt]
  try:
    samples = list(op.sample_inputs("cpu", dtype, requires_grad=False))
  except Exception as e:
    record({"op": tag, "kind": "sample_err", "msg": repr(e)[:200]})
    continue
  for i, s in enumerate(samples[:25]):
    desc = f"{[tuple(t.shape) if isinstance(t, torch.Tensor) else t for t in [s.input] + list(s.args)]} kw={ {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else v) for k, v in s.kwargs.items()} }"[:300]
    try:
      exp = op(s.input, *s.args, **s.kwargs)
    except Exception:
      continue
    try:
      st = s.transform(to_tpu)
      got = op(st.input, *st.args, **st.kwargs)
      got = cpuize(got)
    except Exception as e:
      msg = repr(e).split("\n")[0][:300]
      record({"op": tag, "i": i, "kind": "raise", "msg": msg, "desc": desc})
      continue
    ef, gf = flat(cpuize(exp)), flat(got)
    if len(ef) != len(gf):
      record({"op": tag, "i": i, "kind": "struct", "desc": desc})
      continue
    for j, (a, b) in enumerate(zip(ef, gf)):
      try:
        if name in ("sort", "topk", "argsort", "max", "min", "mode", "kthvalue") and j == 1:
          continue  # index ties
        if a.dtype != b.dtype:
          record({"op": tag, "i": i, "kind": "dtype", "exp": str(a.dtype), "got": str(b.dtype), "desc": desc})
          continue
        if a.shape != b.shape:
          record({"op": tag, "i": i, "kind": "shape", "exp": list(a.shape), "got": list(b.shape), "desc": desc})
          continue
        tol = {} if dt not in ("bf16", "f16") else {"atol": 2e-2, "rtol": 2e-2}
        if dt == "f32":
          tol = {"atol": 1e-4, "rtol": 1e-4}
        torch.testing.assert_close(b, a, equal_nan=True, check_dtype=False, **tol)
      except AssertionError as e:
        m = str(e).split("\n")
        record({"op": tag, "i": i, "out": j, "kind": "value", "msg": " | ".join(m[:4])[:300], "desc": desc})
      except Exception as e:
        record({"op": tag, "i": i, "kind": "cmp_err", "msg": repr(e)[:200]})
record({"op": "__END__"})
