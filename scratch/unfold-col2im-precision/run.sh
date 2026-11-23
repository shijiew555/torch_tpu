# Usage: run.sh <label>. Runs repro and unfold/col2im tests. Use under TPU lock.
E=/root/torch_tpu/.claude/tmp/unfold-col2im-precision-s3
W=/root/torch_tpu/.claude/worktrees/unfold-col2im-precision
PY=$W/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/"
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)
cd $W; export PYTHONPATH=$E/ov
echo "=== $1 repro"; timeout 600 $PY $E/repro.py 2>&1 | grep -v -e '^I0000' -e '^W0000'
echo "=== $1 ops_unit_test filter"
TESTBRIDGE_TEST_ONLY='*col2im* *im2col* *unfold* *fold*' timeout 1800 $PY -m tests.ops_unit_test > $E/tests_$1.log 2>&1; echo "exit $?"
grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED|Max absolute' $E/tests_$1.log
