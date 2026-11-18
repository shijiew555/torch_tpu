import sys, torch, torch_tpu
import transformers as T
sys.path.insert(0, "/root/torch_tpu/.claude/tmp/E1")
from train_eager import build
import accelerate
from accelerate import Accelerator
from accelerate.state import PartialState
try:
    s=PartialState(); print("accelerate PartialState device:", s.device, "distributed_type:", s.distributed_type, flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
try:
    acc=Accelerator(); print("Accelerator device:", acc.device, flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
class DS(torch.utils.data.Dataset):
    def __len__(self): return 16
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(i); ids=torch.randint(0,256,(16,),generator=g)
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids)
args=T.TrainingArguments(output_dir="/root/torch_tpu/.claude/tmp/E1/out",per_device_train_batch_size=4,max_steps=4,logging_steps=1,report_to=[],save_strategy="no",dataloader_pin_memory=False)
print("TrainingArguments.device:", args.device, "n_gpu:", args.n_gpu, flush=True)
m=build("llama")
tr=T.Trainer(model=m,args=args,train_dataset=DS())
r=tr.train()
print("Trainer done; model device:", next(m.parameters()).device, "loss", r.training_loss, flush=True)
