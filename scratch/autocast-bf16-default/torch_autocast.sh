# Runs PyTorch test_autocast.py through the TorchTPU runner. Use under TPU lock.
E=/root/torch_tpu/.claude/tmp/autocast-bf16-default-s3
PY=/root/torch_tpu/.claude/worktrees/autocast-bf16-default/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/"
export TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR=/root/claude-tmp/pytorch/test TORCH_TPU_INTERNAL_TORCH_TEST_MODULE=test_autocast
for N in "$@"; do
  cd $E/repo_$N
  echo "=== $N torch test_autocast"; PYTHONPATH=$E/ov_$N timeout 1200 $PY -m tests.torch.torch_test_runner > $E/torchac_$N.log 2>&1; echo "exit $?"; grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED' $E/torchac_$N.log
done
