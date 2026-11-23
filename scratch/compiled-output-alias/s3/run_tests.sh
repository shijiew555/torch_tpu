#!/bin/bash
# Usage: run_tests.sh <repo> "<overlay tags>" ; TESTS env overrides list.
E=/root/torch_tpu/.claude/tmp/compiled-output-alias3
PY=/root/torch_tpu/.claude/worktrees/compiled-output-alias/.venv/bin/python
cd $E/$1
for t in ${TESTS:-output_alias_test fx_output_reconstructor_test small_graph_test output_modify_test clone_mutated_returned_placeholders_test backend_serialization_test aot_autograd_cache_test cache_test split_compile_test}; do
  for v in $2; do
    rm -rf $E/cache_$v
    L=$E/log_${1}_${v}_$t.txt
    TORCHINDUCTOR_CACHE_DIR=$E/cache_$v PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$E/ov_$v \
      flock /root/tpu.lock timeout 1500 $PY -m tests.compile.$t > $L 2>&1
    echo "$v $t exit=$? $(grep -E '^(Ran|OK|FAILED)' $L | tr '\n' ' ')"
  done
done
