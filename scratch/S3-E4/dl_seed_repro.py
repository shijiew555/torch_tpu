# DataLoader workers call torch.manual_seed(); on TPU that opens the TPU device.
# If the parent has not used the TPU yet, a worker takes the device and the
# parent cannot use the TPU in the training loop.
import faulthandler, sys; faulthandler.dump_traceback_later(120, exit=True)
import torch
import torch_tpu  # noqa: F401

class DS(torch.utils.data.Dataset):
    def __len__(self): return 8
    def __getitem__(self, i): return torch.full((4,), float(i))

if __name__ == "__main__":
    persistent = sys.argv[1] == "persistent"
    model = torch.nn.Linear(4, 1)  # built on CPU, moved later
    dl = torch.utils.data.DataLoader(DS(), batch_size=4, num_workers=2,
                                     persistent_workers=persistent)
    for step, b in enumerate(dl):
        if step == 0:
            model.to("tpu")
        print("step", step, "loss", model(b.to("tpu")).sum().item())
    print("OK")
