# Usage: run.sh <name>... Runs repro and tests for each overlay. Use under TPU lock.
E=/root/torch_tpu/.claude/tmp/autocast-bf16-default-s3
PY=/root/torch_tpu/.claude/worktrees/autocast-bf16-default/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/"
for N in "$@"; do
  cd $E/repo_$N
  echo "=== $N repro"; PYTHONPATH=$E/ov_$N timeout 600 $PY $E/repro.py 2>&1 | grep -v -e '^I0000' -e '^W0000'
  echo "=== $N amp_test"; PYTHONPATH=$E/ov_$N timeout 900 $PY -m tests.amp_test > $E/amp_$N.log 2>&1; echo "exit $?"; grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED' $E/amp_$N.log
  echo "=== $N device_module_tpu_test"; PYTHONPATH=$E/ov_$N timeout 900 $PY -m tests.device.device_module_tpu_test > $E/dev_$N.log 2>&1; echo "exit $?"; grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED' $E/dev_$N.log
done
