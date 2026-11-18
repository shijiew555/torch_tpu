import torch
torch.manual_seed(0)
x = torch.randn(8000).to(torch.bfloat16)
for stable in (False, True):
  i = torch.argsort(x.to("tpu"), stable=stable).cpu()
  v = x[i]
  print("stable", stable, "values sorted:", bool((v[1:] >= v[:-1]).all()),
        "same as cpu stable:", torch.equal(i, torch.argsort(x, stable=True)))
y = torch.randn(8000)
i = torch.argsort(y.to("tpu")).cpu()
print("f32 equal cpu:", torch.equal(i, torch.argsort(y)))
