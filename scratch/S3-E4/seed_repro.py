# Does torch.manual_seed() open the TPU? DataLoader workers call it at start.
import faulthandler; faulthandler.dump_traceback_later(100, exit=True)
import subprocess, sys, torch
import torch_tpu  # noqa: F401
CHILD = "import torch, torch_tpu, sys; print('child: before seed', file=sys.stderr); torch.manual_seed(1); print('child: after seed', file=sys.stderr)"
if __name__ == "__main__":
    print("parent tpu:", torch.ones(2, device="tpu").sum().item(), file=sys.stderr)
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=80)
    err = r.stderr
    print("child rc:", r.returncode, file=sys.stderr)
    print("child stderr has 'Colliding process':", "Colliding process" in err, file=sys.stderr)
    print("child stderr has 'PJRT client not available':", "PJRT client not available" in err, file=sys.stderr)
    for l in err.splitlines():
        if l.startswith("child") or "Colliding" in l or "PJRT client" in l: print("  child>", l[:160], file=sys.stderr)
    r2 = subprocess.run([sys.executable, "-c", "import torch, torch_tpu"], capture_output=True, text=True, timeout=80)
    print("import-only child has 'Colliding process':", "Colliding process" in r2.stderr, file=sys.stderr)
