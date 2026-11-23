# Issue draft

Title: float32 unfold backward and F.fold give bfloat16-accurate results

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

# PR draft

Title: Use HIGHEST precision for float32 unfold backward and col2im

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
