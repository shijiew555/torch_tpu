# torch_tpu PR deliverables

Prepared 2026-11-23.
All branches are on the fork `shijiew555/torch_tpu` and based on upstream `main` at `82d0a494`.
No issue or PR is opened yet. Fill in the GitHub numbers after you open them.

How to submit each one:

1. If the PR has an issue draft, open the issue first: https://github.com/google-pytorch/torch_tpu/issues/new
2. Open the PR with the compare link. Paste the PR title and body.
3. Replace `Fixes #<issue>` in the PR body with the issue number.

## Summary

| # | PR title | Issue first | Branch | Commit | Issue # | PR # |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Make torch.autocast("tpu") default to bfloat16 | Yes | [`autocast-bf16-default-pr2`](https://github.com/shijiew555/torch_tpu/tree/autocast-bf16-default-pr2) | `3ef9ff18` | | |
| 2 | Keep aliasing between outputs of compiled TPU graphs | Yes | [`compiled-output-alias-pr2`](https://github.com/shijiew555/torch_tpu/tree/compiled-output-alias-pr2) | `f7ee0c2f` | | |
| 3 | Restore the TPU RNG state in eager activation checkpointing | Yes | [`ckpt-tpu-rng-state-pr`](https://github.com/shijiew555/torch_tpu/tree/ckpt-tpu-rng-state-pr) | `aae1baaf` | | |
| 4 | Use HIGHEST precision for float32 unfold backward and col2im | Yes | [`unfold-col2im-precision-pr2`](https://github.com/shijiew555/torch_tpu/tree/unfold-col2im-precision-pr2) | `be7d0e6d` | | |
| 5 | Accept sorted=False in torch.unique | No | [`unique-sorted-false-pr2`](https://github.com/shijiew555/torch_tpu/tree/unique-sorted-false-pr2) | `947d7f21` | | |

Notes:

* PR 4 uses HIGHEST precision for float32 (v1). It makes float32 fold and unfold 1.7x to 3.4x slower. If review objects, the fallback (v2) is to follow `torch.set_float32_matmul_precision`, which keeps the default speed.
* PR 5 conflicts with open upstream PR #6879 in the same function. The PR text explains the simple resolution.
* The older branches `autocast-bf16-default-pr` and `compiled-output-alias-pr` are superseded by the `-pr2` branches.

## PR 1: Make torch.autocast("tpu") default to bfloat16

* Branch: https://github.com/shijiew555/torch_tpu/tree/autocast-bf16-default-pr2
* Open PR: https://github.com/google-pytorch/torch_tpu/compare/main...shijiew555:torch_tpu:autocast-bf16-default-pr2?expand=1

### Issue

Title: `torch.autocast("tpu")` without `dtype` runs in float16, not bfloat16

Body:

````markdown
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
````

### PR

Title: Make torch.autocast("tpu") default to bfloat16

Body:

````markdown
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
````

## PR 2: Keep aliasing between outputs of compiled TPU graphs

* Branch: https://github.com/shijiew555/torch_tpu/tree/compiled-output-alias-pr2
* Open PR: https://github.com/google-pytorch/torch_tpu/compare/main...shijiew555:torch_tpu:compiled-output-alias-pr2?expand=1

### Issue

Title: Compiled TPU graphs return copies instead of views when no output requires grad

Body:

````markdown
With `backend="tpu"`, a compiled function that returns a tensor and views of it returns independent copies when no output requires grad.
An in-place write to one output is not seen through the others.
Eager TPU, CPU eager, `aot_eager`, and inductor keep the views.

Repro:

1. Run this script on a TPU host:

   ```python
   import torch
   import torch_tpu  # noqa: F401

   def f(x):
       y = x * 2
       return y, y.view(2, 8), y[1:]

   x = torch.arange(16.0).view(4, 4).to("tpu")
   a, b, c = torch.compile(f, backend="tpu")(x)
   a.add_(100)
   print(b[0, :3].tolist())
   ```

2. Expected: `[100.0, 102.0, 104.0]`, as in eager mode.
3. Actual: `[0.0, 2.0, 4.0]`. `a`, `b`, and `c` do not share storage.

With `x.requires_grad_()` the output is correct, because AOTAutograd regenerates views for outputs that require grad.
For other outputs, AOTAutograd expects the backend to keep the views.

Cause: `_process_fx_outputs` in `src/torch_tpu/_internal/export/export.py` dedups graph outputs only when they are the same view of a buffer.
Every other view becomes a separate graph output with its own buffer.

Proposed fix: return outputs that view the same storage as views of one graph output that covers the full storage.
If no output covers the full storage, add a flat view of the storage as an extra graph output.
Related but different: #6203 (write-back of in-place writes to inputs), #6250 (passthrough elision).

Environment: torch 2.16.0.dev20261112+cpu, libtpu 0.0.53.1, TPU v5e, torch_tpu built from main at afd6943a with Python from main at 82d0a494.
````

### PR

Title: Keep aliasing between outputs of compiled TPU graphs

Body:

````markdown
## Description

Compiled TPU graphs returned independent copies for outputs that are views of each other when no output requires grad, so an in-place write to one output was not seen through the others.
Eager mode, `aot_eager`, and inductor keep these views.

Outputs that view the same storage are now returned as views of one graph output that covers the full storage.
If no returned output covers it, a flat view of the storage is added as an extra graph output.
This also saves one device buffer per aliased output.
Outputs are grouped by storage pointer, and storages with no bytes are not grouped, because `data_ptr()` is 0 for every tensor with no elements.
Graphs without aliased outputs keep the C++ `FxOutputReconstructor` fast path.
Bounded-dynamism graphs and views with a different dtype keep the old behavior.

Fixes #<issue>.

## Test Plan

New test `tests/compile/output_alias_test.py` compiles views, slices, transposes, views without their base, `split`, an empty slice, duplicated outputs, and independent outputs, with and without `requires_grad`.
It checks that output aliasing and the values after an in-place write match CPU eager.
It also checks that the new reconstructor survives pickling, for the compile caches.

Tested commit: f7ee0c2f, on main 82d0a494.
Environment: TPU v5e, torch 2.16.0.dev20261112+cpu, libtpu 0.0.53.1, torch_tpu wheel built at afd6943a, with this branch's Python on `PYTHONPATH`.
The overlay reverts 1694da75 in `compiler.py`, because that commit calls a C++ function that the afd6943a wheel does not have.
`export.py` is unchanged between afd6943a and 82d0a494.

| Test (`python -m tests.compile.<name>`) | main | This branch |
|---|---|---|
| `output_alias_test` (new) | 6 failed, 1 error of 15 | 15 passed |
| `fx_output_reconstructor_test` | 19 passed | 19 passed |
| `small_graph_test` | 75 passed, 3 skipped | 75 passed, 3 skipped |
| `output_modify_test` | 2 passed | 2 passed |
| `clone_mutated_returned_placeholders_test` | 42 passed | 42 passed |
| `backend_serialization_test` | 11 passed, 1 failed | 11 passed, 1 failed |
| `aot_autograd_cache_test` | passed, 5 skipped | passed, 5 skipped |
| `cache_test` | passed, 2 skipped | passed, 2 skipped |
| `split_compile_test` | 18 passed | 18 passed |
| `random_ops_test` | 13 passed | 13 passed |
| `trace_autograd_ops_test` | 4 passed | 4 passed |
| `activation_checkpointing_test` | 14 passed | 14 passed |
| `dtensor_test` | 1 passed | 1 passed |

The `backend_serialization_test` failure is `test_float4_e2m1fn_x2_uses_aot_autograd_cache`, the same on both.
`compile_api_test` aborts the same way on both in `test_force_strides_strided_middle_dimension`, with 61 tests passed before it.
Both happen without this change in this environment.

`BASE_SHA=upstream/main nox -s lint` passes.
````

## PR 3: Restore the TPU RNG state in eager activation checkpointing

* Branch: https://github.com/shijiew555/torch_tpu/tree/ckpt-tpu-rng-state-pr
* Open PR: https://github.com/google-pytorch/torch_tpu/compare/main...shijiew555:torch_tpu:ckpt-tpu-rng-state-pr?expand=1

### Issue

Title: Eager activation checkpointing with dropout gives wrong gradients on TPU

Body:

````markdown
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
````

### PR

Title: Restore the TPU RNG state in eager activation checkpointing

Body:

````markdown
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
````

## PR 4: Use HIGHEST precision for float32 unfold backward and col2im

* Branch: https://github.com/shijiew555/torch_tpu/tree/unfold-col2im-precision-pr2
* Open PR: https://github.com/google-pytorch/torch_tpu/compare/main...shijiew555:torch_tpu:unfold-col2im-precision-pr2?expand=1

### Issue

Title: float32 unfold backward and F.fold give bfloat16-accurate results

Body:

````markdown
`Tensor.unfold` backward, `F.unfold` backward and `F.fold` return float32 results with errors of about 1e-2 on TPU.
The error is the same as when the input is rounded to bfloat16 on CPU.
`torch.set_float32_matmul_precision("highest")` does not change it.
`torch.compile(backend="tpu")` gives the same error.

Repro:

1. Run on a TPU:

   ```python
   import torch
   import torch.nn.functional as F
   import torch_tpu

   torch.manual_seed(0)
   cols = torch.randn(2, 18, 15)
   fold = lambda d: F.fold(cols.to(d), output_size=(5, 6), kernel_size=(3, 2)).cpu()
   print((fold("cpu") - fold("tpu")).abs().max().item())
   ```

2. Expected: an error of about 1e-6.
   Actual: 0.0156.

With the same kind of random input, `F.unfold` backward has an error of 0.0114 and `Tensor.unfold` backward has an error of 0.0072.

Cause: the unfold backward and col2im lowerings use a convolution with 0/1 weights and an empty precision config.
The TPU then computes the convolution with bfloat16 inputs.

Environment: torch 2.16.0.dev20261112+cpu, torch_tpu 0.3.0 built from source at afd6943a, libtpu 0.0.53.1, TPU v5e, Python 3.14.
The root cause lines are unchanged on main at 82d0a494.
````

### PR

Title: Use HIGHEST precision for float32 unfold backward and col2im

Body:

````markdown
## Description

Fixes #<issue>.

`Tensor.unfold` backward, `F.unfold` backward and `F.fold` returned float32 results that were only bfloat16-accurate, with errors of about 1e-2.
Both lowerings sum the input through a convolution with 0/1 weights and an empty precision config, so the TPU rounds the input to bfloat16.
This change sets HIGHEST precision on the input for float32, as `gather.cc` and `scatter.cc` do for their one-hot dots.
The weights are exact 0/1 values, so they keep DEFAULT precision.
Other dtypes do not change.

`test_col2im_fold` now uses the default tolerances.
Its integer inputs never needed the loose tolerances.

## Test Plan

New tests in `tests/ops_unit_test.py` compare float32 results with CPU at the default tolerances:

* `test_col2im_fold_float32_precision`
* `test_im2col_backward_float32_precision`
* `test_unfold_backward_float32_precision`

Results on TPU v5e, with the change applied to afd6943a and built from source:

| Check | Before | After |
| --- | --- | --- |
| `F.fold` max abs error vs CPU, eager and `torch.compile` | 1.6e-2 | 4.8e-7 |
| `F.unfold` backward max abs error vs CPU | 1.1e-2 | 4.8e-7 |
| `Tensor.unfold` backward max abs error vs CPU | 7.2e-3 | 0 |
| `ops_unit_test.py` filter `*col2im* *im2col* *unfold* *fold*` (15 tests) | 3 new tests fail | pass |
| `ops_test.py` filter `*unfold*`, with and without `--compute_grad=True` | pass | pass |

The `ops_unit_test.py` run:

```bash
TESTBRIDGE_TEST_ONLY='*col2im* *im2col* *unfold* *fold*' python -m tests.ops_unit_test
```

HIGHEST precision makes float32 calls slower.
Median steady-state time per call on TPU v5e, batch 8 (Tensor.unfold: 16 x 16000), 3 alternating rounds of before and after:

| Case (float32) | Before | After | Ratio |
| --- | --- | --- | --- |
| `F.fold`, 224x224 image, kernel and stride 16 | 92.1 ms | 309.7 ms | 3.36x |
| `F.unfold` forward and backward, 224x224 image, kernel and stride 16 | 96.4 ms | 314.0 ms | 3.26x |
| `F.fold`, 64 channels, 56x56, kernel 3, padding 1 | 6.1 ms | 11.2 ms | 1.82x |
| `F.unfold` forward and backward, 64 channels, 56x56, kernel 3, padding 1 | 7.1 ms | 12.1 ms | 1.71x |
| `Tensor.unfold` backward, window 400, step 160 | 13.8 ms | 24.8 ms | 1.80x |

The same cases in bfloat16 show no change (ratio 1.00x).

`nox -s lint` and `nox -s format` pass on this branch, which is based on 82d0a494.

Environment: TPU v5e, libtpu 0.0.53.1, torch 2.16.0.dev20261112+cpu, Python 3.14.
````

## PR 5: Accept sorted=False in torch.unique

* Branch: https://github.com/shijiew555/torch_tpu/tree/unique-sorted-false-pr2
* Open PR: https://github.com/google-pytorch/torch_tpu/compare/main...shijiew555:torch_tpu:unique-sorted-false-pr2?expand=1

### Issue

None needed.

### PR

Title: Accept sorted=False in torch.unique

Body:

````markdown
## Description

`torch.unique(x, sorted=False)` raises `unique2(): sorted=False is not yet supported` on TPU.
`sorted=False` lets the op return the unique values in any order, so a sorted result is valid.
The TPU lowering always sorts.
The CUDA `_unique2` kernel also ignores `sorted` and always sorts ([Unique.cu](https://github.com/pytorch/pytorch/blob/e9565720ab4f59a0ed491b3f8440ae38c2455bab/aten/src/ATen/native/cuda/Unique.cu#L200-L207), [UniqueCub.cu](https://github.com/pytorch/pytorch/blob/e9565720ab4f59a0ed491b3f8440ae38c2455bab/aten/src/ATen/native/cuda/UniqueCub.cu#L146-L186)).
The TPU `unique(dim=...)` kernel already ignores `sorted` in the same way.
This change removes the check, so `sorted=False` gives the same sorted result as `sorted=True` and uses the same compiled graphs.

Because of this, `ops_test` no longer needs to skip the `sorted=False` samples of `test_unique`.
Their GPU goldens are sorted, so they now run.
The `max_samples_per_op_dtype=20` override existed only to make up for that skip, so it is removed.
The error test for the old check in `tpu_errors_test.py` is removed.

Repro:

1. Run on a TPU:

   ```python
   import torch
   import torch_tpu

   x = torch.tensor([3, 1, 3, 2, 1])
   print(torch.unique(x.to("tpu"), sorted=False).cpu())
   ```

2. Expected: `tensor([1, 2, 3])`, as on CPU and CUDA.
   Actual: `RuntimeError: unique2(): sorted=False is not yet supported`.

Open PR #6879 changes the same `AtenUnique2` kernel and conflicts textually with this change.
The resolution is simple: #6879 keeps the same `TT_CHECK_THROW`, so this change then only removes that check.
The `ops_unit_test.py` conflict is two new tests added at the same place.

## Test Plan

`test_unique_unsorted` in `tests/ops_unit_test.py` compares TPU `sorted=False` results with CPU `sorted=True` results for float32, bfloat16, int64 and bool, with all combinations of `return_inverse` and `return_counts`.

Results on TPU v5e, before and after the change.
The tests ran from the repo tree against a torch_tpu wheel built at afd6943a, with and without the change.
They ran as absltest mains, for example `TESTBRIDGE_TEST_ONLY='*unique*' python -m tests.ops_unit_test`.
`ops_test` used the pinned GPU goldens from `tests/goldens.pin` and the "before" run already had the skip removed.

| Test | Before | After |
| --- | --- | --- |
| Repro above | RuntimeError | `tensor([1, 2, 3])` |
| `ops_unit_test`, filter `*unique*` (791 tests) | 16 failures, all in `test_unique_unsorted` | OK (72 skipped, same as before) |
| `ops_test` `test_unique` and `test_unique_consecutive`, max 20 samples | 30 subtest failures, all `sorted=False is not yet supported` | OK (190 subtests) |
| Same, max 5 samples (OSS setting) | not run | OK (50 subtests) |
| `tpu_errors_test` (full, 368 tests) | not run | 1 failure, `test_pallas_mesh_invariants_errors`: `jax` is not installed in the test venv, not related |
| Removed `test_unique2_unsupported_sorted` | passes | fails, as expected without the check |

`nox -s format` made no changes, and `BASE_SHA=upstream/main nox -s lint` passed on the PR branch.

Environment: TPU v5e (1 chip), torch 2.16.0.dev20261112+cpu, libtpu 0.0.53.1, Python 3.14.
torch_tpu was built from afd6943a with the change, because a build at 82d0a494 needs a full XLA rebuild.
The PR branch is the same change on 82d0a494.
The only merge conflict was in the `test_unique` skip condition of `ops_test.py`, which upstream had changed.
````
