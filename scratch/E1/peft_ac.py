import copy, time, torch, torch_tpu, traceback
import transformers as T
from train_eager import build, batch
def tryit(label, fn):
    try:
        t=time.time(); r=fn(); print(f"{label}: OK {r} ({time.time()-t:.1f}s)", flush=True)
    except Exception as e:
        traceback.print_exc(); print(f"{label}: FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
def lora():
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0); m=build("llama")
    cfg=LoraConfig(r=8,lora_alpha=16,target_modules=["q_proj","v_proj"],lora_dropout=0.0)
    mc=get_peft_model(copy.deepcopy(m),cfg); torch.manual_seed(0); mt=get_peft_model(copy.deepcopy(m),cfg)
    mt.load_state_dict(mc.state_dict()); mt=mt.to("tpu")
    res=[]
    for mm,dev in [(mc,"cpu"),(mt,"tpu")]:
        opt=torch.optim.AdamW([p for p in mm.parameters() if p.requires_grad],lr=1e-2)
        ls=[]
        for _ in range(3):
            o=mm(**batch("llama",dev)); o.loss.backward(); opt.step(); opt.zero_grad(); ls.append(round(o.loss.item(),5))
        res.append(ls)
    merged=mt.merge_and_unload()
    return f"cpu={res[0]} tpu={res[1]} merged_dev={next(merged.parameters()).device}"
def autocast_compile():
    torch.manual_seed(0); m=build("llama"); mt=copy.deepcopy(m).to("tpu"); f=torch.compile(mt)
    with torch.autocast("cpu",dtype=torch.bfloat16): lc=m(**batch("llama","cpu")).loss
    with torch.autocast("tpu",dtype=torch.bfloat16): lt=f(**batch("llama","tpu")).loss
    lt.backward()
    return f"loss cpu={lc.item():.4f} tpu={lt.item():.4f}"
BACKEND=None
def trainer_bf16_compile():
    class DS(torch.utils.data.Dataset):
        def __len__(self): return 16
        def __getitem__(self,i):
            g=torch.Generator().manual_seed(i); ids=torch.randint(0,256,(16,),generator=g)
            return dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids)
    args=T.TrainingArguments(output_dir="/root/torch_tpu/.claude/tmp/E1/out",per_device_train_batch_size=4,max_steps=4,logging_steps=4,report_to=[],save_strategy="no",bf16=True,torch_compile=True,torch_compile_backend=BACKEND,eval_strategy="no")
    tr=T.Trainer(model=build("llama"),args=args,train_dataset=DS(),eval_dataset=DS())
    r=tr.train(); ev=tr.evaluate()
    return f"train_loss={r.training_loss:.4f} eval_loss={ev['eval_loss']:.4f}"
if __name__=="__main__":
    import sys
    if len(sys.argv)>1: BACKEND=sys.argv[1]
    for t in (sys.argv[2].split(",") if len(sys.argv)>2 else ["lora","autocast_compile","trainer_bf16_compile"]): tryit(t, globals()[t])
