# Issue draft

## Title

Eager activation checkpointing with dropout gives wrong gradients on TPU

## Body

Eager `torch.utils.checkpoint` gives wrong gradients on TPU when the checkpointed function uses dropout.
There is no error and no warning.
This affects every eager model that uses activation checkpointing, for example Hugging Face `gradient_checkpointing_enable()`.
`torch.compile` is not affected.

Repro:

1. Run this script on a TPU host:

   ```python
   import torch
   import torch_tpu
   from torch import nn
   import torch.utils.checkpoint as cp

   for dev in ("cpu", "tpu"):
       for reentrant in (True, False):
           torch.manual_seed(0)
           lin = nn.Linear(8, 8).to(dev)
           drop = nn.Dropout(0.5)
           f = lambda x: drop(lin(x))
           x = torch.randn(4, 8, device=dev, requires_grad=True)
           torch.manual_seed(1)
           cp.checkpoint(f, x, use_reentrant=reentrant).sum().backward()
           g_ckpt = x.grad.clone()
           x.grad = None
           torch.manual_seed(1)
           f(x).sum().backward()
           print(dev, reentrant, (g_ckpt - x.grad).abs().max().item())
   ```

2. Expected: `0.0` on all four lines, as on CPU.
   Actual: `0.0` on CPU, `1.5137939453125` on TPU for both `use_reentrant` values.

Cause: `torch.utils.checkpoint` saves the device RNG state in the forward pass only if `getattr(device_module, "_initialized", False)` is true.
`torch.cuda` has this attribute.
`torch.tpu` does not, so checkpoint never saves the TPU RNG state, and the recompute in backward draws a new dropout mask.

Proposed fix: add `torch.tpu._initialized`, true when the TPU runtime is initialized.

Environment: TPU v5e, torch 2.16.0.dev20261112+cpu, torch_tpu 0.3.0 built from `afd6943a` with the Python sources of `82d0a494`, libtpu 0.0.53.1.

# PR draft

## Title

Restore the TPU RNG state in eager activation checkpointing

## Body

## Description

Eager `torch.utils.checkpoint` gave wrong gradients on TPU when the checkpointed function used dropout.
Checkpoint saves and restores the device RNG state only if the device module has a true `_initialized` attribute.
`torch.cuda` defines it, but `torch.tpu` did not.
So the recompute in backward drew a new dropout mask.

This change adds `torch.tpu._initialized`, as a property on the device module metaclass.
It uses the same check as `torch.tpu.is_initialized()`, but without the experimental-API warning, because checkpoint reads it on every call.
`torch.utils.checkpoint` is the only reader of `_initialized` in PyTorch.

Fixes #<issue>

## Test Plan

`test_eager_activation_checkpointing_dropout` in `tests/rng_cuda_ref_test.py` compares the gradients of a `Linear` and `Dropout` function with and without checkpointing, for both `use_reentrant` values.
`test_initialized` in the device module tests also checks `torch.tpu._initialized`.

| Check | Before | After |
| --- | --- | --- |
| Repro script, max grad difference on TPU | 1.51 | 0.0 |
| Two `TransformerEncoderLayer(dropout=0.1)` layers, relative grad difference | 0.705 | 0 |
| `rng_cuda_ref_test`, new cases (2) | fail | pass |
| `rng_cuda_ref_test` (70) | - | pass, 1 skipped |
| `rng_test` (21) | - | pass |
| `device_module_test_tpu` (41) | `test_initialized` fails | pass |
| `public_api_test`, `annotations_test` | - | pass |
| `BASE_SHA=upstream/main nox -s lint` | - | pass |

I did not build this commit with Bazel.
I ran the tests as absltest mains with the torch_tpu 0.3.0 wheel built from `afd6943a` and a `PYTHONPATH` overlay of the Python sources.
The new tests ran on the sources of this commit (base `82d0a494`).
The full `rng_cuda_ref_test` and `rng_test` ran on the `afd6943a` sources with this change applied, because 11 compiled RNG tests need a C++ symbol that the `afd6943a` wheel does not have.

Environment: TPU v5e, torch 2.16.0.dev20261112+cpu, libtpu 0.0.53.1, Python 3.14.
