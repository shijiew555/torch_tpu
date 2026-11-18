import time, torch
from torch.testing._internal.common_methods_invocations import op_db
ops = [o for o in op_db if o.name == "norm"]
for op in ops:
  for i, s in enumerate(op.sample_inputs("cpu", torch.int64)):
    try:
      e = op(s.input, *s.args, **s.kwargs)
    except Exception as ex:
      print(op.variant_test_name, i, "cpu raises", type(ex).__name__); continue
    t = time.time()
    st = s.transform(lambda x: x.to("tpu") if isinstance(x, torch.Tensor) else x)
    print(op.variant_test_name, i, s.input.shape, s.args, s.kwargs, flush=True)
    try:
      g = op(st.input, *st.args, **st.kwargs).cpu()
      print("  ok", time.time()-t, e, g, flush=True)
    except Exception as ex:
      print("  tpu raises", repr(ex)[:200], flush=True)
