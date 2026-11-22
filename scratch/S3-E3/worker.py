"""Runs op_db error_inputs for given ops on CPU and TPU; writes JSON lines."""
import json, sys, traceback, faulthandler
import torch
from torch.testing._internal.common_methods_invocations import op_db

faulthandler.enable()
OUT = sys.argv[1]
KEYS = set(sys.argv[2:])
DEV = "tpu"


def mat(x):
  if isinstance(x, torch.Tensor):
    return x.cpu()
  if isinstance(x, (list, tuple)):
    return [mat(y) for y in x]
  return x


def short(x):
  if isinstance(x, torch.Tensor):
    return f"T{tuple(x.shape)}:{x.dtype}"
  if isinstance(x, (list, tuple)):
    return "[" + ",".join(short(y) for y in x[:4]) + "]"
  return repr(x)[:60]


def run(op, si):
  try:
    r = op(si.input, *si.args, **si.kwargs)
    r = mat(r)
    if DEV in str(si.input) or True:
      try:
        torch.accelerator.synchronize()
      except Exception:
        pass
    return ("ok", short(r))
  except Exception as e:
    return (type(e).__name__, str(e)[:400])


f = open(OUT, "a")
for op in op_db:
  key = op.name + "|" + op.variant_test_name
  if key not in KEYS or op.error_inputs_func is None:
    continue
  try:
    cpu_eis = list(op.error_inputs("cpu"))
  except Exception as e:
    f.write(json.dumps({"op": key, "gen_err_cpu": repr(e)[:200]}) + "\n")
    f.flush()
    continue
  try:
    tpu_eis = list(op.error_inputs(DEV))
  except Exception as e:
    f.write(json.dumps({"op": key, "gen_err_tpu": repr(e)[:300]}) + "\n")
    f.flush()
    continue
  used = set()
  pairs = []
  for c in cpu_eis:
    t = None
    for j, cand in enumerate(tpu_eis):
      if j not in used and str(cand.error_regex) == str(c.error_regex) and cand.error_type == c.error_type:
        used.add(j); t = cand; break
    pairs.append((c, t))
  for i, (c, t) in enumerate(pairs):
    moved = t is None
    if moved:
      def mv(x):
        if isinstance(x, torch.Tensor): return x.to(DEV)
        if isinstance(x, list): return [mv(y) for y in x]
        if isinstance(x, tuple): return tuple(mv(y) for y in x)
        if isinstance(x, dict): return {k: mv(v) for k, v in x.items()}
        if isinstance(x, torch.device) or (isinstance(x, str) and x == "cpu"): return torch.device(DEV)
        return x
      cs = c.sample_input
      t = type(c)(type(cs)(mv(cs.input), args=mv(cs.args), kwargs=mv(cs.kwargs)), error_type=c.error_type, error_regex=c.error_regex)
    desc = f"in={short(c.sample_input.input)} args={short(c.sample_input.args)} kw={short(c.sample_input.kwargs)}"
    f.write(json.dumps({"op": key, "i": i, "start": True, "desc": desc}) + "\n")
    f.flush()
    cr = run(op, c.sample_input)
    tr = run(op, t.sample_input)
    f.write(json.dumps({"op": key, "i": i, "desc": desc,
                        "expect": [c.error_type.__name__, str(c.error_regex)],
                        "cpu": cr, "tpu": tr, "moved": moved}) + "\n")
    f.flush()
f.write(json.dumps({"done": True}) + "\n")
