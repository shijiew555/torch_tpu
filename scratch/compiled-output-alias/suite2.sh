#!/bin/bash
E=/root/torch_tpu/.claude/tmp/compiled-output-alias
W=/root/torch_tpu/.claude/worktrees/compiled-output-alias
DST=$E/ov/torch_tpu/_internal/export/export.py
$E/sync.sh
git -C $W show upstream/main:src/torch_tpu/_internal/export/export.py > $DST
$E/run.sh 900 -m tests.compile.output_alias_test > $E/final_before_output_alias_test.log 2>&1
echo "before output_alias_test exit=$? $(grep -E '^(Ran|OK|FAILED)' $E/final_before_output_alias_test.log | tr '\n' ' ')"
cp $W/src/torch_tpu/_internal/export/export.py $DST
for t in output_alias_test random_ops_test trace_autograd_ops_test compile_api_test activation_checkpointing_test dtensor_test; do
  $E/run.sh 1800 -m tests.compile.$t > $E/final_after_$t.log 2>&1
  echo "after $t exit=$? $(grep -E '^(Ran|OK|FAILED)' $E/final_after_$t.log | tr '\n' ' ')"
done
