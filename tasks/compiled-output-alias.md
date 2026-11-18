# compiled-output-alias (worker 3)

* 07:20Z Worktree at upstream/main 3c6acc43. Repro on overlay confirms bug (backend=tpu no-grad row: shared=False). Log: .claude/tmp/compiled-output-alias/repro_before.log
* Plan: Python-only. In `_process_fx_outputs`, group outputs by storage; emit one flat storage-wide output per group; rebuild members with `as_strided` in a Python reconstructor that wraps the C++ one. Graphs without aliased outputs keep the C++ fast path.
* New test: tests/compile/output_alias_test.py
* 07:20Z New test fails before fix: 5 of 12 cases (views+slice, transpose, views-only, split no-grad, split grad). Log test_before.log.
* 07:22Z First fix (flat storage base for all groups) broke training graphs: output y became a view made inside the autograd Function, so in-place writes then reads raise. Fixed: when an output covers the full storage, it is the base (plain tensor); a hidden flat base is used only for views-only groups. Test now reads outputs with autograd checks (mutant check: forced flat base makes 3 grad cases ERROR).
* 07:23Z Commit e1d54824 pushed. Existing compile test suite before/after running.
* 07:40Z Existing compile tests (9 modules) before/after: all pass, no change. New test: before 5/12 fail, after 12/12 pass. Lint passes (commit 9858e789). Review pass done, no further changes. Final validation (final test file before/after + 5 more compile modules after) running.
* 08:09Z Done. Backup `compiled-output-alias` 9244adbe. PR branch `compiled-output-alias-pr` 0fc40a74 (squash 424a80b4 + pickle-test fix 0fc40a74; no force-push). Lint passes on both.
* Final: new test 13/13 after (before: 5 fail + 1 error). Existing 9 modules unchanged. compile_api_test: 2 optimization_barrier tests fail before and after (overlay env, not this change).
* Draft: tasks/compiled-output-alias-pr.md (issue + PR).
