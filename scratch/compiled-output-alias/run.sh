#!/bin/bash
# Usage: run.sh <secs> <python args...>; runs from repo copy with overlay and a fresh inductor cache.
E=/root/torch_tpu/.claude/tmp/compiled-output-alias
secs=$1; shift
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /root/claude-tmp/ind_cache.XXXX)
cd $E/repo && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$E/ov flock /root/tpu.lock timeout $secs python3 "$@"
