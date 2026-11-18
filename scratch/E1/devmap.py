import torch, torch_tpu, accelerate
import transformers as T
from train_eager import build
d="/root/torch_tpu/.claude/tmp/E1/saved"; build("llama").save_pretrained(d)
for dm in ["auto", "tpu", {"": 0}]:
    try:
        m=T.AutoModelForCausalLM.from_pretrained(d, device_map=dm)
        print("accelerate", accelerate.__version__, "device_map", dm, "->", next(m.parameters()).device, flush=True)
    except Exception as e: print("device_map", dm, "FAIL", type(e).__name__, str(e)[:200], flush=True)
