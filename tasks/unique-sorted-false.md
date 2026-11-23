# unique-sorted-false

## Session 3 refresh (in progress)

* Test branch unique-sorted-false-s3 = backup commits cherry-picked on
  afd6943a, plus one commit that removes the sorted=False skip of
  ops_test test_unique (GPU goldens: CUDA _unique2 always sorts).
  Worktree .claude/worktrees/unique-sorted-false, venv: torch
  2.16.0.dev20261112+cpu, libtpu 0.0.53.1, afd6943a wheel + Python overlay.
* Repro before (afd6943a wheel): CPU torch.unique(x, sorted=False) works;
  TPU raises "unique2(): sorted=False is not yet supported".
* PR #6879 (open upstream) conflicts textually with our -pr branch in
  unique_aten_kernels.cc and ops_unit_test.py (git merge-tree). Trivial
  resolution: #6879 keeps the same TT_CHECK_THROW.
* Note: `flock /root/bazel.lock bazel` leaks the lock to the bazel server.
  Use `flock -o`.
* Wheel build of -s3 running.

## Session 2

State: building test target (before fix). Started 08:12Z.

* Worktree: .claude/worktrees/unique-sorted-false, base 3c6acc43.
* Test added: tests/ops_unit_test.py test_unique_unsorted (TPU sorted=False vs CPU sorted=True).
* Plan: run test before fix (expect fail), then remove TT_CHECK_THROW in AtenUnique2,
  remove tpu_errors_test.py test_unique2_unsupported_sorted.
* Build log: tmp/unique-sorted-false/build_before.log
