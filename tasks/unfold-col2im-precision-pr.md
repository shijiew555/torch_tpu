# Issue draft

Title: float32 unfold backward and F.fold give bfloat16-accurate results

`Tensor.unfold` backward, `F.unfold` backward and `F.fold` return float32 results with errors of about 1e-2 on TPU.
The error is the same as when the input is rounded to bfloat16 on CPU.
`torch.set_float32_matmul_precision("highest")` does not change it.

Repro:

1. Run on a TPU:

   ```python
   import torch
   import torch.nn.functional as F

   torch.manual_seed(0)
   cols = torch.randn(2, 18, 15)
   fold = lambda d: F.fold(cols.to(d), output_size=(5, 6), kernel_size=(3, 2)).cpu()
   print((fold("cpu") - fold("tpu")).abs().max().item())
   ```

2. Expected: an error of about 1e-6.
   Actual: 0.0156.

Cause: the unfold backward and col2im lowerings use a convolution with 0/1 weights and an empty precision config.
The TPU then computes the convolution with bfloat16 inputs.

Environment: torch 2.14.1+cpu, torch_tpu 0.3.0, libtpu 0.0.50, TPU v5e.

# PR draft

Title: Use HIGHEST precision for float32 unfold backward and col2im

## Description

Fixes #<issue>.

`Tensor.unfold` backward, `F.unfold` backward and `F.fold` returned float32 results that were only bfloat16-accurate, with errors of about 1e-2.
Both lowerings sum the input through a convolution with 0/1 weights and an empty precision config, so the TPU rounds the input to bfloat16.
This change sets HIGHEST precision on the input for float32, as `gather.cc` does for its one-hot dot.
The weights are exact 0/1 values, so they keep DEFAULT precision.

`test_col2im_fold` now uses the default tolerances.
Its integer inputs never needed the loose tolerances.

## Test Plan

New tests in `tests/ops_unit_test.py` compare float32 results with CPU at the default tolerances:

* `test_col2im_fold_float32_precision`
* `test_im2col_backward_float32_precision`
* `test_unfold_backward_float32_precision`

Before the fix, the three new tests fail with maximum absolute errors of 1.6e-2 for `F.fold`, 1.3e-2 for `F.unfold` backward and 5.6e-3 for `Tensor.unfold` backward.
After the fix, all im2col, col2im and unfold tests in `ops_unit_test.py` pass:

```bash
bazel test --config=wheel_common --config=no_rbe \
  --spawn_strategy=sandboxed,local --strategy=sandboxed,local \
  --//:test_wheel=True --local_test_jobs=1 \
  '--test_filter=*col2im* *im2col* *unfold*' \
  //tests:ops_unit_test_tpu-v5lite
```

`nox -s lint` passes.

Environment: TPU v5e, libtpu 0.0.50, torch 2.14.1+cpu, hermetic Python 3.14.
