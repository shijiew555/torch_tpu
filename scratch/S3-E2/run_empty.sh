for c in long_empty_idx float_empty_idx empty_2d_rows index_select nonempty_empty_idx bool_mask_then_index; do
  echo "== $c"; timeout 120 /root/torch_tpu/.claude/worktrees/build-base/.venv/bin/python empty_index.py $c 2>&1 | rg -v "^I0000|^WARNING|tcmalloc|Successfully|Registered|Device type|Initializing" | rg "tensor|Error|error|SIGABRT|stablehlo" | head -4
done
