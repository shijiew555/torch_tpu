E=/root/torch_tpu/.claude/tmp/autocast-bf16-default-s3
PY=/root/torch_tpu/.claude/worktrees/autocast-bf16-default/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/"
export TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR=/root/claude-tmp/pytorch/test TORCH_TPU_INTERNAL_TORCH_TEST_MODULE=test_ops
for N in "$@"; do
  cd $E/repo_$N
  echo "=== $N test_ops -k test_fake_autocast"; TESTBRIDGE_TEST_ONLY='*test_fake_autocast_[j-z]* *test_fake_autocast_i[o-z]* *test_fake_autocast_in[a-ce-z]* *test_fake_autocast_index_[s-z]* *test_fake_autocast_indices*' PYTHONPATH=$E/ov_$N timeout 2400 $PY -m tests.torch.torch_test_runner > $E/fakeac2_$N.log 2>&1; echo "exit $?"; grep -E '^Ran |^OK|^FAILED' $E/fakeac2_$N.log; grep -cE '^(FAIL|ERROR):' $E/fakeac2_$N.log
done
