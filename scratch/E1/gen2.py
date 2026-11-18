import sys, copy, time, torch, torch_tpu, traceback
from archs import mk, nodrop
if len(sys.argv)>2: torch.set_float32_matmul_precision(sys.argv[2])
for name in sys.argv[1].split(","):
    try:
        torch.manual_seed(0); m,kind=mk(name); m.eval(); mt=copy.deepcopy(m).to("tpu")
        g=torch.Generator().manual_seed(1)
        if kind=="whisper":
            b=dict(input_features=torch.randn(2,16,64,generator=g))
        else:
            ids=torch.randint(3,256,(2,10),generator=g); mask=torch.ones_like(ids); mask[1,:3]=0; ids[1,:3]=0
            b=dict(input_ids=ids,attention_mask=mask)
        kw=dict(max_new_tokens=20,do_sample=False,pad_token_id=0,eos_token_id=None)
        oc=m.generate(**b,**kw)
        t=time.time(); ot=mt.generate(**{k:v.to("tpu") for k,v in b.items()},**kw); t1=time.time()-t
        print(f"{name} generate match={torch.equal(oc,ot.cpu())} t={t1:.1f}s", flush=True)
        if not torch.equal(oc,ot.cpu()): print(" cpu",oc.tolist(),"\n tpu",ot.cpu().tolist())
    except Exception as e:
        traceback.print_exc(); print(f"{name} generate FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
