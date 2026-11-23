# Paired ABAB benchmark of before/after wheels. Run under one TPU lock.
E=/root/torch_tpu/.claude/tmp/unfold-col2im-precision-s3
PY=/root/torch_tpu/.claude/worktrees/unfold-col2im-precision/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/"
cd /tmp
for round in 1 2 3; do
  for L in before after; do
    PYTHONPATH=$E/whl_$L timeout 900 $PY $E/bench.py $L 2>&1 | grep '^{' | sed "s/}$/, \"round\": $round}/"
  done
done
