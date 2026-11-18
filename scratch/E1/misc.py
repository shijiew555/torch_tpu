import sys, copy, time, torch, torch_tpu, os
import transformers as T
sys.path.insert(0, "/root/torch_tpu/.claude/tmp/E1")
from train_eager import build, batch
def tryit(label, fn):
    try:
        t=time.time(); r=fn(); print(f"{label}: OK {r} ({time.time()-t:.1f}s)", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{label}: FAIL {type(e).__name__}: {str(e)[:400]}", flush=True)
tests=sys.argv[1].split(",")
torch.manual_seed(0)
base=build("llama").eval()
g=torch.Generator().manual_seed(1)
ids=torch.randint(3,256,(2,10),generator=g); mask=torch.ones_like(ids); mask[1,:3]=0; ids[1,:3]=0
ref=base.generate(input_ids=ids,attention_mask=mask,max_new_tokens=12,do_sample=False,pad_token_id=0,eos_token_id=None)
def static_cache(compile_):
    m=copy.deepcopy(base).to("tpu")
    if compile_: m.forward=torch.compile(m.forward)
    out=m.generate(input_ids=ids.to("tpu"),attention_mask=mask.to("tpu"),max_new_tokens=12,do_sample=False,pad_token_id=0,eos_token_id=None,cache_implementation="static")
    return f"match={torch.equal(out.cpu(),ref)}"
def attn_impl(impl):
    torch.manual_seed(0); m=build("llama"); m.config._attn_implementation=impl
    m=m.to("tpu"); m.set_attn_implementation(impl) if hasattr(m,"set_attn_implementation") else None
    mc=copy.deepcopy(m).cpu()
    o=m(**batch("llama","tpu")); o.loss.backward()
    oc=mc(**batch("llama","cpu")); oc.loss.backward()
    return f"loss tpu={o.loss.item():.5f} cpu={oc.loss.item():.5f}"
def grad_ckpt():
    torch.manual_seed(0); m=build("llama").to("tpu"); m.gradient_checkpointing_enable(); m.train()
    o=m(**batch("llama","tpu")); o.loss.backward()
    return f"loss={o.loss.item():.5f} gnorm={torch.nn.utils.clip_grad_norm_(m.parameters(),1.0).item():.4f}"
def long_seq(L):
    torch.manual_seed(0); m=build("llama"); m.config.max_position_embeddings=4096; m=T.LlamaForCausalLM(m.config)
    mt=copy.deepcopy(m).to("tpu")
    x=torch.randint(0,256,(1,L)); 
    oc=m(input_ids=x,labels=x); ot=mt(input_ids=x.to("tpu"),labels=x.to("tpu")); ot.loss.backward()
    return f"L={L} loss cpu={oc.loss.item():.5f} tpu={ot.loss.item():.5f} maxlogitdiff={(oc.logits-ot.logits.cpu()).abs().max().item():.2e}"
def pipeline_():
    m=copy.deepcopy(base)
    tok=None
    p=T.pipeline("feature-extraction", model=m, tokenizer=None, device="tpu")
    return str(p.device)
def save_load():
    m=copy.deepcopy(base).to("tpu"); d="/root/torch_tpu/.claude/tmp/E1/saved"
    m.save_pretrained(d); m2=T.LlamaForCausalLM.from_pretrained(d, device_map="tpu")
    return f"dev={next(m2.parameters()).device} same={all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(m.parameters(),m2.parameters()))}"
def ignore_index():
    torch.manual_seed(0); m=build("llama"); mt=copy.deepcopy(m).to("tpu")
    b=batch("llama","cpu"); b["labels"][:, :8]=-100; b["labels"][1,:]=-100
    oc=m(**b); ot=mt(**{k:v.to("tpu") for k,v in b.items()})
    return f"loss cpu={oc.loss.item():.5f} tpu={ot.loss.item():.5f}"
def bf16_generate():
    m=copy.deepcopy(base).to(torch.bfloat16); mt=copy.deepcopy(m).to("tpu")
    oc=m.generate(input_ids=ids,attention_mask=mask,max_new_tokens=12,do_sample=False,pad_token_id=0,eos_token_id=None)
    ot=mt.generate(input_ids=ids.to("tpu"),attention_mask=mask.to("tpu"),max_new_tokens=12,do_sample=False,pad_token_id=0,eos_token_id=None)
    return f"match={torch.equal(oc,ot.cpu())}"
def device_map_load():
    d="/root/torch_tpu/.claude/tmp/E1/saved"; base.save_pretrained(d)
    m2=T.AutoModelForCausalLM.from_pretrained(d, device_map="auto")
    return f"dev={next(m2.parameters()).device}"
fns=dict(static=lambda:static_cache(False), static_compile=lambda:static_cache(True), sdpa=lambda:attn_impl("sdpa"), eager_attn=lambda:attn_impl("eager"), flex=lambda:attn_impl("flex_attention"),
  ckpt=grad_ckpt, long1024=lambda:long_seq(1024), long2048=lambda:long_seq(2048), pipeline=pipeline_, saveload=save_load, ignore=ignore_index, bf16gen=bf16_generate, devmap_auto=device_map_load)
for t in tests: tryit(t, fns[t])
