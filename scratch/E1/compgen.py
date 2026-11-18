import copy, time, torch, torch_tpu
import transformers as T
from train_eager import build
torch.manual_seed(0)
base=build("llama").eval()
g=torch.Generator().manual_seed(1)
ids=torch.randint(3,256,(2,10),generator=g); mask=torch.ones_like(ids); mask[1,:3]=0; ids[1,:3]=0
ref=base.generate(input_ids=ids,attention_mask=mask,max_new_tokens=16,do_sample=False,pad_token_id=0,eos_token_id=None)
m=copy.deepcopy(base).to("tpu"); m.forward=torch.compile(m.forward)
for i in range(2):
    t=time.time()
    out=m.generate(input_ids=ids.to("tpu"),attention_mask=mask.to("tpu"),max_new_tokens=16,do_sample=False,pad_token_id=0,eos_token_id=None)
    print(f"compiled dynamic-cache generate run{i} match={torch.equal(out.cpu(),ref)} t={time.time()-t:.1f}s", flush=True)
import torch._dynamo.utils as du
print("dynamo counters:", dict(du.counters["stats"]), dict(du.counters["recompiles"]) if "recompiles" in du.counters else "")
