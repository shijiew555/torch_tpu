# Usage: opstest.sh <label>. OpInfo ops_test (TPU vs GPU goldens) for unfold ops. Use under TPU lock.
E=/root/torch_tpu/.claude/tmp/unfold-col2im-precision-s3
W=/root/torch_tpu/.claude/worktrees/unfold-col2im-precision
cd $W
export PYTHONPATH=$E/ov PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/"
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)
for G in False True; do
  TESTBRIDGE_TEST_ONLY='*unfold*' timeout 2400 .venv/bin/python -m tests.ops_test --golden_data_base_dir=$E/goldens --compute_grad=$G > $E/opstest_$1_grad$G.log 2>&1
  echo "=== $1 ops_test compute_grad=$G exit $?"
  grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED' $E/opstest_$1_grad$G.log
done
