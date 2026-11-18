import torch, torch_tpu
import torch._dynamo.utils as du
emb = torch.nn.Embedding(256, 64).to("tpu")
def f(ids, labels):
    h = emb(ids)
    return torch.nn.functional.cross_entropy(h.view(-1, 64), labels.view(-1))
cf = torch.compile(f)
for L in [16, 20, 24, 28]:
    x = torch.randint(0, 64, (2, L)).to("tpu")
    print(L, cf(x, x).item(), "unique_graphs", du.counters["stats"]["unique_graphs"], flush=True)
def g(ids):
    return ids * 2
cg = torch.compile(g)
for L in [16, 20, 24, 28]:
    print("g", L, cg(torch.ones(2, L, device="tpu")).sum().item(), "unique_graphs", du.counters["stats"]["unique_graphs"], flush=True)
