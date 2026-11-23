bash /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/run.sh after /root/torch_tpu/.claude/worktrees/unique-sorted-false
bash /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/run_err.sh after_oldtest /root/torch_tpu/.claude/worktrees/build-base
bash /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/run_ops.sh after20 /root/torch_tpu/.claude/worktrees/unique-sorted-false 20
bash /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/run_ops.sh after5 /root/torch_tpu/.claude/worktrees/unique-sorted-false 5
cd /root/torch_tpu/.claude/worktrees/unique-sorted-false; export PYTHONPATH=/root/torch_tpu/.claude/tmp/unique-sorted-false-s3/ov PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/"
echo "=== after full tpu_errors_test"
TORCH_SHOW_CPP_STACKTRACES=0 TORCH_TPU_ENABLE_DEBUG_CHECKS=1 timeout 2400 /root/torch_tpu/.claude/worktrees/unique-sorted-false/.venv/bin/python -m tests.tpu_errors_test --test_mode=tpu > /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/errors_full_after.log 2>&1; echo "exit $?"
grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED' /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/errors_full_after.log
