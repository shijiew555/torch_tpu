import copy, torch, torch_tpu
import transformers as T
from train_eager import build
torch.manual_seed(0)
base=build("llama").eval()
g=torch.Generator().manual_seed(1)
ids=torch.randint(3,256,(2,10),generator=g); mask=torch.ones_like(ids); mask[1,:3]=0; ids[1,:3]=0
m=copy.deepcopy(base).to(torch.bfloat16); mt=copy.deepcopy(m).to("tpu")
kw=dict(max_new_tokens=12,do_sample=False,pad_token_id=0,eos_token_id=None,output_scores=True,return_dict_in_generate=True)
oc=m.generate(input_ids=ids,attention_mask=mask,**kw)
ot=mt.generate(input_ids=ids.to("tpu"),attention_mask=mask.to("tpu"),**kw)
print("cpu",oc.sequences[:,10:].tolist()); print("tpu",ot.sequences[:,10:].cpu().tolist())
for step,(sc,st) in enumerate(zip(oc.scores,ot.scores)):
    st=st.cpu().float(); sc=sc.float()
    top=sc.topk(2,dim=-1)
    print(step,"maxdiff=%.3e"%(sc-st).abs().max().item(),"cpu top2 gap",(top.values[:,0]-top.values[:,1]).tolist())
    if not torch.equal(sc.argmax(-1),st.argmax(-1)): print("  first divergence"); break
# same-prefix check: single forward on identical inputs
x=torch.randint(3,256,(2,10),generator=g)
lc=m(input_ids=x).logits.float(); lt=mt(input_ids=x.to("tpu")).logits.cpu().float()
print("single fwd bf16 logits maxdiff", (lc-lt).abs().max().item(), "max|logit|", lc.abs().max().item())
