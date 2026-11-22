# DataLoader with worker processes on a TPU host (Python 3.14, default start method).
import faulthandler, sys; faulthandler.dump_traceback_later(120, exit=True)
import multiprocessing, torch
import torch_tpu  # noqa: F401

class DS(torch.utils.data.Dataset):
    def __len__(self): return 8
    def __getitem__(self, i): return torch.full((4,), float(i))

if __name__ == "__main__":
    print("start method:", multiprocessing.get_start_method())
    if sys.argv[1] == "tpu_first":
        print("parent tpu:", torch.ones(2, device="tpu").sum().item())
    dl = torch.utils.data.DataLoader(DS(), batch_size=4, num_workers=2)
    tot = sum(b.sum().item() for b in dl)
    print("data total:", tot)
    print("tpu after loader:", torch.ones(2, device="tpu").sum().item())
