import sys, torch, torch_tpu
import transformers as T
from train_eager import build
class DS(torch.utils.data.Dataset):
    def __len__(self): return 16
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(i); ids=torch.randint(0,256,(16,),generator=g)
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids)
args=T.TrainingArguments(output_dir="/root/torch_tpu/.claude/tmp/E1/out",per_device_train_batch_size=4,max_steps=4,logging_steps=1,report_to=[],save_strategy="no",bf16=len(sys.argv)>1)
print("TrainingArguments.device:", args.device, "n_gpu:", args.n_gpu, "accelerate state device:", args.distributed_state.device, flush=True)
m=build("llama")
tr=T.Trainer(model=m,args=args,train_dataset=DS())
try:
    r=tr.train(); print("Trainer OK loss", r.training_loss, flush=True)
except Exception as e: print("Trainer FAIL", type(e).__name__, str(e)[:200], flush=True)
