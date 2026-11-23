# Usage: run.sh <label> <errors_test_dir>. Use under the TPU lock.
E=/root/torch_tpu/.claude/tmp/unique-sorted-false-s3
W=/root/torch_tpu/.claude/worktrees/unique-sorted-false
PY=$W/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/"
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)
export PYTHONPATH=$E/ov
$PY -c "import torch, importlib.metadata as m; print('torch', torch.__version__, 'libtpu', m.version('libtpu'))" 2>/dev/null
cd $W
echo "=== $1 repro"; timeout 600 $PY $E/repro.py 2>&1 | grep -v -e '^I0000' -e '^W0000' -e 'WARNING: All log' -e '^Successfully\|^Registered\|^Device type\|^Initializing'
echo "=== $1 ops_unit_test '*unique*'"
TESTBRIDGE_TEST_ONLY='*unique*' timeout 2400 $PY -m tests.ops_unit_test > $E/unit_$1.log 2>&1; echo "exit $?"
grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED' $E/unit_$1.log
echo "=== $1 tpu_errors_test '*unique*' from $2"
cd $2; TORCH_SHOW_CPP_STACKTRACES=0 TORCH_TPU_ENABLE_DEBUG_CHECKS=1 TESTBRIDGE_TEST_ONLY="*unique*" timeout 1200 $PY -m tests.tpu_errors_test --test_mode=tpu > $E/errors_$1.log 2>&1; echo "exit $?"
grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED' $E/errors_$1.log
