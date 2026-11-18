import torch, torch_tpu
print("torch.tpu._get_autocast_dtype():", torch.tpu._get_autocast_dtype())
print("torch.get_autocast_dtype('tpu'):", torch.get_autocast_dtype("tpu"))
a = torch.randn(4, 4, device="tpu"); b = torch.randn(4, 4, device="tpu")
with torch.autocast("tpu"):
    print("mm dtype under torch.autocast('tpu'):", (a @ b).dtype)
with torch.autocast("cpu"):
    print("mm dtype under torch.autocast('cpu'):", (a.cpu() @ b.cpu()).dtype)
# fp16 overflow that bf16 would survive
x = torch.full((4, 4), 300.0, device="tpu")
with torch.autocast("tpu"):
    print("300x300 sum of 4 under autocast('tpu'):", (x @ x).float().cpu()[0, 0].item())
