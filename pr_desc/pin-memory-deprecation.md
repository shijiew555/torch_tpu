# PR draft

Title: Stop deprecation warnings on non-blocking copies from unpinned memory

## Description

A non-blocking copy of an unpinned CPU tensor to TPU emitted two `DeprecationWarning`s with torch 2.14.
Test suites that turn warnings into errors failed on this copy.
The copy pins the source first, and it passed the destination device to `pin_memory()`.
torch 2.14 deprecated the device argument of `pin_memory()` and `is_pinned()`.
This change calls `pin_memory()` without an argument.
Memory pinned this way is TPU-pinned, because TorchTPU registers the TPU as the accelerator.
The tests also stop passing the deprecated device argument.

Repro:

1. On a TPU, run with `python -W error::DeprecationWarning`:

   ```python
   import torch
   import torch_tpu

   torch.ones(4).to("tpu", non_blocking=True)
   ```

2. Expected: no warning.
   Actual: `DeprecationWarning: The argument 'device' of Tensor.pin_memory() is deprecated.`

## Test Plan

RESULTS_PLACEHOLDER

Environment: TPU v5e, libtpu 0.0.50, torch 2.14.1+cpu, hermetic Python 3.14.
