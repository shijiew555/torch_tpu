set -x
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/root/torch_tpu/.claude/tmp/E6/overlay_main
python -c "import torch_tpu; print(torch_tpu.__file__)"
python /root/torch_tpu/.claude/tmp/E4/repro_stream.py
python /root/torch_tpu/.claude/tmp/E4/repro_pin_warn.py
python /root/torch_tpu/.claude/tmp/E4/repro_pin_warn_default.py
