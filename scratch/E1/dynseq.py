import copy, time, torch, torch_tpu
import torch._dynamo.utils as du
from train_eager import build
torch.manual_seed(0)
m=build("llama"); mt=copy.deepcopy(m).to("tpu"); f=torch.compile(mt)
for L in [16,20,24,28,32,17,20]:
    x=torch.randint(0,256,(2,L),generator=torch.Generator().manual_seed(L)); mask=torch.ones_like(x); mask[0,:3]=0
    t=time.time(); lt=f(input_ids=x.to("tpu"),attention_mask=mask.to("tpu"),labels=x.to("tpu")).loss; lt.backward(); v=lt.item(); dt=time.time()-t
    lc=m(input_ids=x,attention_mask=mask,labels=x).loss
    mt.zero_grad()
    print(f"L={L} loss cpu={lc.item():.5f} tpu={v:.5f} t={dt:.1f}s unique_graphs={du.counters['stats']['unique_graphs']}", flush=True)
