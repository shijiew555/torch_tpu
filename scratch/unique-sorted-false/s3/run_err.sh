# Usage: run_err.sh <label> <tests_root>. Use under the TPU lock.
E=/root/torch_tpu/.claude/tmp/unique-sorted-false-s3
PY=/root/torch_tpu/.claude/worktrees/unique-sorted-false/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/" PYTHONPATH=$E/ov
echo "=== $1 tpu_errors_test '*unique*' from $2"
cd $2; TORCH_SHOW_CPP_STACKTRACES=0 TORCH_TPU_ENABLE_DEBUG_CHECKS=1 TESTBRIDGE_TEST_ONLY="*unique*" timeout 1200 $PY -m tests.tpu_errors_test --test_mode=tpu > $E/errors_$1.log 2>&1; echo "exit $?"
grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED' $E/errors_$1.log
