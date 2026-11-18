import sys, copy, time, torch, torch_tpu
import transformers as T
torch.manual_seed(0)
def build(name):
    if name=="llama":
        c=T.LlamaConfig(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
        return T.LlamaForCausalLM(c)
    if name=="gpt2":
        c=T.GPT2Config(vocab_size=256,n_embd=64,n_layer=2,n_head=4,n_positions=128)
        return T.GPT2LMHeadModel(c)
    if name=="bert":
        c=T.BertConfig(vocab_size=256,hidden_size=64,num_hidden_layers=2,num_attention_heads=4,intermediate_size=128,max_position_embeddings=128)
        return T.BertForMaskedLM(c)
    if name=="t5":
        c=T.T5Config(vocab_size=256,d_model=64,d_kv=16,d_ff=128,num_layers=2,num_heads=4,decoder_start_token_id=0,dropout_rate=0.0)
        return T.T5ForConditionalGeneration(c)
    if name=="qwen2":
        c=T.Qwen2Config(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
        return T.Qwen2ForCausalLM(c)
    if name=="gemma":
        c=T.GemmaConfig(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=128)
        return T.GemmaForCausalLM(c)
def batch(name, dev):
    g=torch.Generator().manual_seed(1)
    ids=torch.randint(0,256,(2,16),generator=g)
    mask=torch.ones_like(ids); mask[1,:4]=0
    b=dict(input_ids=ids,attention_mask=mask,labels=ids.clone())
    if name=="t5": b["labels"]=torch.randint(0,256,(2,8),generator=g)
    return {k:v.to(dev) for k,v in b.items()}
def run(m, name, dev, steps, compile_):
    m=m.to(dev); m.train()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout): mod.p=0.0
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    fwd = torch.compile(m) if compile_ else m
    losses=[]
    for s in range(steps):
        out=fwd(**batch(name,dev)); out.loss.backward(); opt.step(); opt.zero_grad()
        losses.append(out.loss.item())
    return losses, out.logits.detach().cpu().float()
if __name__=='__main__':
  names=sys.argv[1].split(","); comp=len(sys.argv)>2 and sys.argv[2]=="compile"
  for name in names:
      try:
          torch.manual_seed(0); m=build(name); m2=copy.deepcopy(m)
          lc,oc=run(m,name,"cpu",3,False)
          t=time.time(); lt,ot=run(m2,name,"tpu",3,comp)
          print(f"{name} compile={comp} cpu={['%.5f'%x for x in lc]} tpu={['%.5f'%x for x in lt]} maxlogitdiff={(oc-ot).abs().max().item():.2e} t={time.time()-t:.1f}s", flush=True)
      except Exception as e:
          import traceback; traceback.print_exc(); print(f"{name} FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
