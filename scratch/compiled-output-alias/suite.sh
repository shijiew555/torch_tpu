#!/bin/bash
# Runs existing compile tests on upstream export.py (before) and fixed export.py (after).
E=/root/torch_tpu/.claude/tmp/compiled-output-alias
W=/root/torch_tpu/.claude/worktrees/compiled-output-alias
DST=$E/ov/torch_tpu/_internal/export/export.py
TESTS="fx_output_reconstructor_test output_modify_test small_graph_test clone_mutated_returned_placeholders_test backend_serialization_test aot_autograd_cache_test cache_test split_compile_test output_alias_test"
for phase in before after; do
  if [ $phase = before ]; then git -C $W show upstream/main:src/torch_tpu/_internal/export/export.py > $DST; else cp $W/src/torch_tpu/_internal/export/export.py $DST; fi
  for t in $TESTS; do
    $E/run.sh 1500 -m tests.compile.$t > $E/suite_${phase}_$t.log 2>&1
    echo "$phase $t exit=$? $(grep -E '^(Ran|OK|FAILED)' $E/suite_${phase}_$t.log | tr '\n' ' ')"
  done
done
