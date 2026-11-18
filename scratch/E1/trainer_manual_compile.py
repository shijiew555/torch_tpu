import torch, torch_tpu
import transformers as T
from train_eager import build
class DS(torch.utils.data.Dataset):
    def __len__(self): return 16
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(i); ids=torch.randint(0,256,(16,),generator=g)
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids)
args=T.TrainingArguments(output_dir="/root/torch_tpu/.claude/tmp/E1/out",per_device_train_batch_size=4,max_steps=4,logging_steps=4,report_to=[],save_strategy="no",bf16=True)
m=build("llama"); m.compile()
tr=T.Trainer(model=m,args=args,train_dataset=DS(),eval_dataset=DS())
r=tr.train(); ev=tr.evaluate()
print(f"manual compile Trainer OK train_loss={r.training_loss:.4f} eval_loss={ev['eval_loss']:.4f}")
