# unique-sorted-false (worker 4)

State: building test target (before fix). Started 08:12Z.

* Worktree: .claude/worktrees/unique-sorted-false, base 3c6acc43.
* Test added: tests/ops_unit_test.py test_unique_unsorted (TPU sorted=False vs CPU sorted=True).
* Plan: run test before fix (expect fail), then remove TT_CHECK_THROW in AtenUnique2,
  remove tpu_errors_test.py test_unique2_unsupported_sorted.
* Build log: tmp/unique-sorted-false/build_before.log
