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

Environment: TPU v5e, torch 2.16.0.dev20261112+cpu, torch_tpu 0.3.0 built from `afd6943a`, libtpu 0.0.53.1.

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

Two upstream tests, `TestAutocastDevice.test_cache_disabled` and `test_cast_cache_is_global`, count casts of a weight to float16.
With the bfloat16 default the weight is cast to bfloat16, so they fail, and this change excludes them for TPU.

Fixes #<issue>

## Test Plan

New tests in `amp_test` check `torch.get_autocast_dtype("tpu")`, the output dtype and overflow of a matmul under `torch.autocast("tpu")`, and the dtype seen in autograd backward.
They fail before the change and pass after it.

| Test | Before | After |
| --- | --- | --- |
| Repro script above | float16, inf | bfloat16, 360448.0 |
| New `amp_test` tests (3) | 3 fail | pass |
| `device_module_test_tpu` (41) | pass | pass |
| PyTorch `test_autocast` | 7 pass, 2 skipped | 7 pass, 2 skipped, 2 excluded |
| PyTorch `test_fake_autocast` without `index_reduce` | 4 errors | same 4 errors |

`test_autocast_rnn` fails before and after in my environment with `failed to inline functions in MLIR module`.
It needs the C++ change in 5bf70a27, which the binaries I used do not have.
The `index_reduce` cases of `test_fake_autocast` abort the process before and after the change, so I ran them separately from the rest.

Environment: TPU v5e, torch 2.16.0.dev20261112+cpu, libtpu 0.0.53.1.
The tests ran with the Python sources of this branch at 3ef9ff18 on top of torch_tpu binaries built from `afd6943a`, not with a Bazel build of this branch.
PyTorch tests came from the matching PyTorch commit `82b208e2`.

```
python -m tests.amp_test
python -m tests.device.device_module_tpu_test
TORCH_TPU_INTERNAL_TORCH_TEST_MODULE=test_autocast python -m tests.torch.torch_test_runner
BASE_SHA=upstream/main nox -s lint
```
