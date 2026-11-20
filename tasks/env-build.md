# Environment build (new Colab VM, 2026-11-20)

Status: ready.
The restored Bazel cache works on a new VM.

## Results

* `pip install torch_tpu==0.3.0` into system Python 3.13: done in 10 s (was 0.2.0).
* TPU smoke test with the installed 0.3.0 wheel: passed, `tensor([2., 2., 2., 2.])`, 13 s.
* Worktree: `/root/torch_tpu/.claude/worktrees/build-base`, detached at upstream `3c6acc43`.
* Wheel build (flags from `BUILD_NOTES.md`): passed in 716 s (about 12 min), including analysis.
  Summary: 20,617 total actions, 13,713 disk cache hits, 6,904 internal, 0 local compiles.
* Wheel: `/root/torch_tpu/.claude/worktrees/build-base/bazel-bin/ci/wheel/torch_tpu-0.3.0-cp314-cp314-manylinux_2_31_x86_64.whl` (107 MB).
* Test targets built with `--//:test_wheel=True` in 138 s:
  `//tests:ops_unit_test_tpu-v5lite`, `//tests:tpu_errors_test`, `//tests:async_transfer_test`, `//tests/stream:copy_test`.
  All four names exist at this commit.
* Smoke test `//tests:async_transfer_test` on TPU under `/root/tpu.lock`: PASSED in 28 s (32 s wall).

## Notes

* Logs: `/root/torch_tpu/.claude/tmp/env-build/` (`build.log`, `build_tests.log`, `smoke_test.log`).
* The build log has many `Could not find a whl or an sdist with hash` warnings from rules_python.
  They are harmless and the build passed.
