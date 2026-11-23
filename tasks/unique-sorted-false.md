# unique-sorted-false

## Session 3 refresh (done)

State: done, validated. PR draft updated. Nothing posted.

* Branches (origin): unique-sorted-false-s3 78bc93f8 (test branch on
  afd6943a), unique-sorted-false-pr2 947d7f21 (one commit on upstream/main
  82d0a494). Old branches untouched.
* New in s3: ops_test test_unique no longer skips sorted=False samples
  (GPU goldens are sorted; CUDA _unique2 always sorts, Unique.cu L200-207
  at pytorch e9565720). The max_samples_per_op_dtype=20 override (only for
  that skip) is removed.
* Merge onto 82d0a494: one conflict in the ops_test skip_if (upstream
  191c30aff moved the compiled-mode skip to skipTest). Resolved, same
  meaning.
* PR #6879 (open) conflicts textually with -pr2 in unique_aten_kernels.cc
  (it keeps the same TT_CHECK_THROW) and ops_unit_test.py (adjacent new
  tests). Mentioned in the draft.
* Build: wheel of s3, 1879 s, 13603 disk cache hits. Wheel copy:
  tmp/unique-sorted-false-s3/. Error string gone from the binaries.
* Results (TPU v5e, torch 2.16.0.dev20261112+cpu, libtpu 0.0.53.1):
  * Repro: before RuntimeError, after tensor([1, 2, 3]).
  * ops_unit_test '*unique*' (791): before 16 FAIL (test_unique_unsorted),
    after OK (72 skipped).
  * ops_test TestOps.test_unique* with goldens: before 30 subtest FAIL,
    after OK (20 samples: 190 subtests; 5 samples: 50 subtests).
  * Old test_unique2_unsupported_sorted: before OK, after FAIL (expected).
  * Full tpu_errors_test after: 368 run, 1 FAIL
    test_pallas_mesh_invariants_errors (jax not installed in venv).
  * nox -s format: no change. BASE_SHA=upstream/main nox -s lint: pass.
* Logs: scratch/unique-sorted-false/s3/.
* Lock note: `flock /root/bazel.lock bazel` leaks the lock to the bazel
  server. Use `flock -o`.

## Session 2

State: building test target (before fix). Started 08:12Z.

* Worktree: .claude/worktrees/unique-sorted-false, base 3c6acc43.
* Test added: tests/ops_unit_test.py test_unique_unsorted (TPU sorted=False vs CPU sorted=True).
* Plan: run test before fix (expect fail), then remove TT_CHECK_THROW in AtenUnique2,
  remove tpu_errors_test.py test_unique2_unsupported_sorted.
* Build log: tmp/unique-sorted-false/build_before.log
