E=/root/torch_tpu/.claude/tmp/autocast-bf16-default-s3
PY=/root/torch_tpu/.claude/worktrees/autocast-bf16-default/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/"
echo "=== new tests (pr2 tests) on base_main source"
cd $E/repo_fix_pr2 && PYTHONPATH=$E/ov_base_main timeout 900 $PY -m tests.amp_test > $E/amp_newtests_on_base.log 2>&1; echo "exit $?"; grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED' $E/amp_newtests_on_base.log
echo "=== device_module (pr2 tests) on base_main source"
PYTHONPATH=$E/ov_base_main timeout 900 $PY -m tests.device.device_module_tpu_test > $E/dev_newtests_on_base.log 2>&1; echo "exit $?"; grep -E '^(FAIL|ERROR):|^Ran |^OK|^FAILED' $E/dev_newtests_on_base.log
bash $E/run.sh fix_pr2
bash $E/torch_autocast.sh fix_pr2
