import torch, torch_tpu, tempfile, traceback, copy
from safetensors.torch import save_file, load_file, save
m = torch.nn.Linear(4,4); m(torch.ones(2,4)).sum().backward()
print("cpu deepcopy grad kept:", copy.deepcopy(m).weight.grad is not None, flush=True)
x = torch.arange(12.).reshape(3,4); xt = x.to("tpu")
p = tempfile.mktemp()
for name, f in [("save_file tpu", lambda: save_file({"w": xt}, p)),
                ("save bytes tpu", lambda: len(save({"w": xt}))),
                ("load_file device=tpu", lambda: load_file((save_file({"w": x}, p), p)[1], device="tpu")["w"].device),
                ("load_file cpu then .to", lambda: load_file(p)["w"].to("tpu").device)]:
    try: print("OK", name, f(), flush=True)
    except Exception: print("FAIL", name); traceback.print_exc()
