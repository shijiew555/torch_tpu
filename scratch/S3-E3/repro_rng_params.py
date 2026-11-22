"""In-place RNG ops with invalid scalar parameters: CPU vs TPU."""
import torch

CASES = {
  "uniform_(3, -1)": lambda x: x.uniform_(3, -1),
  "uniform_(0, inf)": lambda x: x.uniform_(0, float("inf")),
  "exponential_(0)": lambda x: x.exponential_(0),
  "exponential_(-1)": lambda x: x.exponential_(-1),
  "normal_(0, -1)": lambda x: x.normal_(0, -1),
  "cauchy_(0, -1)": lambda x: x.cauchy_(0, -1),
  "log_normal_(0, -1)": lambda x: x.log_normal_(0, -1),
  "geometric_(0)": lambda x: x.geometric_(0),
  "geometric_(1.5)": lambda x: x.geometric_(1.5),
  "bernoulli_(1.5)": lambda x: x.bernoulli_(1.5),
  "random_(5, 2)": lambda x: x.random_(5, 2),
  "nn.init.uniform_(a=1,b=0)": lambda x: torch.nn.init.uniform_(x, 1, 0),
  "nn.init.trunc_normal_(std=-1)": lambda x: torch.nn.init.trunc_normal_(x, std=-1.0),
}
for n, f in CASES.items():
  out = []
  for dev in ["cpu", "tpu"]:
    x = torch.zeros(5, device=dev)
    try:
      r = f(x).cpu()
      out.append(f"{dev}: ok {r.tolist()}")
    except Exception as e:
      out.append(f"{dev}: {type(e).__name__}: " + str(e).replace("\n", " | ")[:160])
  print(n, "\n  " + "\n  ".join(out), flush=True)
