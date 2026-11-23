# PR draft

Branch: `unique-sorted-false-pr2` (one commit on upstream/main 82d0a494).

Title: Accept sorted=False in torch.unique

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
