"""Runs worker.py over op chunks, each under the TPU lock, restarts after crashes."""
import json, subprocess, sys
P = "/root/torch_tpu/.claude/worktrees/build-base/.venv/bin/python"
keys = json.load(open("ops.json"))
out = sys.argv[1]
CH = int(sys.argv[2]) if len(sys.argv) > 2 else 40
i = 0
while i < len(keys):
  chunk = keys[i:i + CH]
  open(out, "a").write(json.dumps({"chunk": chunk[0]}) + "\n")
  r = subprocess.run(["flock", "/root/tpu.lock", "timeout", "600", P,
                      "worker.py", out] + chunk,
                     capture_output=True, text=True)
  lines = [json.loads(l) for l in open(out)]
  if r.returncode != 0:
    last = [l for l in lines if "op" in l][-1] if lines else None
    open(out, "a").write(json.dumps({"crash": r.returncode, "last": last,
                                     "stderr": r.stderr[-1500:]}) + "\n")
    # Skip past the op that crashed.
    done = {l["op"] for l in lines if "op" in l}
    if last and last["op"] in chunk:
      i = i + chunk.index(last["op"]) + 1
      continue
  i += CH
print("finished")
