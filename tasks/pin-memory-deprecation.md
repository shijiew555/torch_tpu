# pin-memory-deprecation (worker 5)

State: before-fix build running (started 08:13Z).

* Base: upstream/main 3c6acc43. Worktree: .claude/worktrees/pin-memory-deprecation.
* Backup branch pushed: pin-memory-deprecation db9de1b4 (test only).
* Fix plan: `pin_memory()` without device arg in csrc/ops/copy_from/cpu_to_tpu.cc.
  No `is_pinned(device)` call in csrc; the is_pinned warning comes from
  inside torch's pin_memory(device).
* Test: async_transfer_test.test_to_tpu_non_blocking_unpinned_source
  (simplefilter("error") + set_warn_always). Also removed deprecated
  pin_memory(device) in async_transfer_test.py and stream/copy_test.py.
* Targets: //tests:async_transfer_test //tests/stream:copy_test.
