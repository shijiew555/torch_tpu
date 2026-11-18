import sys, copy, time, torch, torch_tpu
import transformers as T
sys.path.insert(0, "/root/torch_tpu/.claude/tmp/E1")
from train_eager import build
names=sys.argv[1].split(",")
for name in names:
    torch.manual_seed(0); m=build(name).eval(); mt=copy.deepcopy(m).to("tpu")
    g=torch.Generator().manual_seed(1)
    ids=torch.randint(3,256,(2,10),generator=g)
    mask=torch.ones_like(ids); mask[1,:3]=0; ids[1,:3]=0
    for label,kw in [("greedy",dict(do_sample=False)),("greedy_nocache",dict(do_sample=False,use_cache=False)),("beam",dict(do_sample=False,num_beams=3))]:
        try:
            oc=m.generate(input_ids=ids,attention_mask=mask,max_new_tokens=20,pad_token_id=0,eos_token_id=None,**kw)
            t=time.time()
            ot=mt.generate(input_ids=ids.to("tpu"),attention_mask=mask.to("tpu"),max_new_tokens=20,pad_token_id=0,eos_token_id=None,**kw)
            t1=time.time()-t; t=time.time()
            ot=mt.generate(input_ids=ids.to("tpu"),attention_mask=mask.to("tpu"),max_new_tokens=20,pad_token_id=0,eos_token_id=None,**kw)
            t2=time.time()-t
            ot=ot.cpu()
            print(f"{name} {label} match={torch.equal(oc,ot)} t1={t1:.1f}s t2={t2:.1f}s", flush=True)
            if not torch.equal(oc,ot): print(" cpu",oc.tolist(),"\n tpu",ot.tolist())
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"{name} {label} FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
    try:
        torch.manual_seed(0)
        ot=mt.generate(input_ids=ids.to("tpu"),attention_mask=mask.to("tpu"),max_new_tokens=20,pad_token_id=0,do_sample=True,top_k=50,top_p=0.9,temperature=0.8)
        print(f"{name} sample ok shape={tuple(ot.shape)} range=({ot.min().item()},{ot.max().item()})", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} sample FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
