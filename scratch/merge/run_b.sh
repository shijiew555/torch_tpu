M=/root/torch_tpu/.claude/tmp/merge
run() { name=$1; shift; echo "== $name"; flock /root/tpu.lock timeout 300 "$@" > $M/$name.log 2>&1; echo "exit=$?"; grep -vE "^I0000|^W0000|^num_|^peak|^$" $M/$name.log | tail -8; }
cd /root/torch_tpu/.claude/tmp/E1 && run autocast python autocast_default.py
cd /root/torch_tpu/.claude/tmp/E4 && run stream python repro_stream.py
cd /root/torch_tpu/.claude/tmp/E4 && run pinwarn python repro_pin_warn.py
cd /root/torch_tpu/.claude/tmp/E2 && run alias python alias_repro.py
