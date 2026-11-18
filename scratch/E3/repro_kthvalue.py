import torch
x = torch.tensor([[5., 1., 4.], [2., 8., 3.]])
print("cpu:", torch.kthvalue(x, 2, dim=1))
print("tpu:", torch.kthvalue(x.to("tpu"), 2, dim=1))
