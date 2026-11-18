import torch
torch.manual_seed(0)
x = torch.randn(100000, dtype=torch.float64) * 3
xt = x.float().to("tpu")
def rel(name, f):
  ref = f(x)
  t = f(xt).cpu().double()
  c = f(x.float()).double()
  print(f"{name:12s} tpu max rel err {((t-ref).abs()/ref.abs().clamp_min(1e-30)).max().item():.2e}  cpu f32 {((c-ref).abs()/ref.abs().clamp_min(1e-30)).max().item():.2e}")
rel("exp", torch.exp)
rel("log", lambda v: torch.log(v.abs() + 0.5))
rel("log1p", lambda v: torch.log1p(v.abs()))
rel("logsumexp", lambda v: torch.logsumexp(v.reshape(1000, 100), 1))
rel("lcse", lambda v: torch.logcumsumexp(v.reshape(1000, 100), 1))
rel("lcse_dim0_5", lambda v: torch.logcumsumexp(v[:125].reshape(5, 5, 5), 0))
rel("cumsum", lambda v: torch.cumsum(v.reshape(1000, 100), 1))
