"""Missing sampler ops for torch.distributions on TPU."""
import torch
import torch_tpu  # noqa: F401

D = torch.distributions
a = torch.full((3,), 2.0, device="tpu")
for name, f in [
    ("Gamma", lambda: D.Gamma(a, 1.0).sample()),
    ("Beta", lambda: D.Beta(a, 3.0).sample()),
    ("Dirichlet", lambda: D.Dirichlet(a).sample()),
    ("Poisson", lambda: D.Poisson(a).sample()),
    ("Binomial", lambda: D.Binomial(10, a / 4).sample()),
]:
  try:
    print(name, f().cpu())
  except RuntimeError as e:
    print(name, "FAIL:", e)
