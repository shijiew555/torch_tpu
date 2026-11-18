# Issue draft

## Title

`torch.autocast("tpu")` without `dtype` runs in float16, not bfloat16

## Body

`torch.autocast("tpu")` without a `dtype` argument runs ops in float16.
The docs say it uses bfloat16 by default on TPU, and `torch.tpu._get_autocast_dtype()` returns bfloat16.
Code ported from CUDA often omits `dtype`, so it silently runs in float16 and can overflow.

Repro:

1. Run this script on a TPU host:

   ```python
   import torch
   import torch_tpu

   print(torch.get_autocast_dtype("tpu"))
   x = torch.full((4, 4), 300.0, device="tpu")
   with torch.autocast("tpu"):
       y = x @ x
   print(y.dtype, y[0, 0].item())
   ```

Expected: `torch.bfloat16`, then `torch.bfloat16 360448.0`.

Actual: `torch.float16`, then `torch.float16 inf`.

Cause: `torch.autocast` reads its default from `torch.get_autocast_dtype("tpu")`.
PyTorch core sets this default to float16 for PrivateUse1 backends.
TorchTPU only sets a Python class attribute and never calls `torch.set_autocast_dtype`.

Proposed fix: call `torch.set_autocast_dtype("tpu", torch.bfloat16)` when TorchTPU initializes the device.
PyTorch keeps this value per thread.
The call covers the thread that loads TorchTPU and autograd backward, which inherits the caller's autocast state.
A Python thread started later still gets float16.
A full fix for all threads needs a per-backend default in PyTorch core.

Environment: TPU v5e, torch 2.14.1+cpu, torch_tpu 0.3.0, libtpu 0.0.50.

# PR draft

## Title

Make torch.autocast("tpu") default to bfloat16

## Body

## Description

`torch.autocast("tpu")` without `dtype` runs in float16, because PyTorch core defaults the PrivateUse1 autocast dtype to float16 and TorchTPU never overrides it.
This contradicts the docs and `torch.tpu._get_autocast_dtype()`, which both say bfloat16, and it lets code ported from CUDA overflow.
TorchTPU now sets the default to bfloat16 when it initializes the device.

PyTorch keeps this default per thread, so the change covers the thread that loads TorchTPU and autograd backward.
Threads that the user starts later still default to float16 until PyTorch core supports a per-backend default.
The API reference now documents this.

Fixes #<issue>

## Test Plan

New tests in `amp_test` check `torch.get_autocast_dtype("tpu")`, the output dtype and overflow of a matmul under `torch.autocast("tpu")`, and the dtype seen in autograd backward.
They fail before the change and pass after it.
`device_module_test_tpu` passes before and after.
`test_autocast_works` fails before and after in my environment, because the installed 0.3.0 binaries predate the `aten::linear` lowering change that its expected MLIR checks.

Environment: TPU v5e, torch 2.14.1+cpu, libtpu 0.0.50.
The tests ran with this branch's Python sources on top of the torch_tpu 0.3.0 binaries, not with a Bazel build.

```
python3 -m tests.amp_test
python3 -m tests.device.device_module_tpu_test
BASE_SHA=upstream/main nox -s lint
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01PrgHFEYJoJ26r8bKhDx9RT
