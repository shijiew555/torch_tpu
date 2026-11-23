# Usage: run_ops.sh <label> <tests_dir> <max_samples>. Use under the TPU lock.
E=/root/torch_tpu/.claude/tmp/unique-sorted-false-s3
PY=/root/torch_tpu/.claude/worktrees/unique-sorted-false/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/"
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)
export PYTHONPATH=$E/ov
cd $2
echo "=== $1 ops_test test_unique (dir $2, max_samples $3)"
TESTBRIDGE_TEST_ONLY='TestOps.test_unique' timeout 2400 $PY -m tests.ops_test --golden_data_base_dir=$E/goldens --max_samples_per_op_dtype=$3 --test_categories=-foreach,-nonzero,-searchsorted > $E/ops_$1.log 2>&1; echo "exit $?"
grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED|sorted=False|skipped' $E/ops_$1.log | head -20
