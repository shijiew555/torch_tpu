# PR draft

Title: Accept sorted=False in torch.unique

## Description

`torch.unique(x, sorted=False)` raised `unique2(): sorted=False is not yet supported` on TPU.
`sorted=False` lets the op return the unique values in any order, so a sorted result is valid.
The TPU lowering always sorts, and the CUDA `_unique2` kernel and the TPU `unique(dim=...)` kernel already ignore `sorted=False` in the same way.
This change removes the check, so `sorted=False` returns the same sorted result as `sorted=True`.

Repro:

1. Run on a TPU:

   ```python
   import torch

   x = torch.tensor([3, 1, 3, 2, 1])
   print(torch.unique(x.to("tpu"), sorted=False).cpu())
   ```

2. Expected: `tensor([1, 2, 3])`.
   Actual: `RuntimeError: unique2(): sorted=False is not yet supported`.

The error test for the old check in `tpu_errors_test.py` is removed.

## Test Plan

`test_unique_unsorted` in `tests/ops_unit_test.py` compares TPU `sorted=False` results with CPU `sorted=True` results for float32, bfloat16, int64 and bool, with all combinations of `return_inverse` and `return_counts`.

RESULTS_PLACEHOLDER

Environment: TPU v5e, libtpu 0.0.50, torch 2.14.1+cpu, hermetic Python 3.14.
