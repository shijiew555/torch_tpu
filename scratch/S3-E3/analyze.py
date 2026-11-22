import json, re, sys
BAD = re.compile(r"INTERNAL|Check failed|\.cc:|\.h:|stablehlo|mhlo|'func\.|verif|unknown error|UNKNOWN|XlaRuntimeError|libtpu|MLIR|absl", re.I)
for l in open(sys.argv[1]):
  d = json.loads(l)
  if "crash" in d:
    print("CRASH", d["crash"], d["last"] and d["last"].get("op"), d["last"] and d["last"].get("desc"), d["stderr"][-600:].replace("\n"," | ")); continue
  if "gen_err_tpu" in d or "gen_err_cpu" in d:
    print("GEN", d); continue
  if "cpu" not in d: continue
  c, t = d["cpu"], d["tpu"]
  if c[0] == "ok": continue  # CPU did not raise; skip
  cls = None
  if t[0] == "ok": cls = "c"
  elif BAD.search(t[1]): cls = "d"
  elif t[0] != c[0]: cls = "e"
  if cls:
    print(f"[{cls}] {d['op']} #{d['i']} moved={d['moved']} {d['desc'][:120]}\n    cpu: {c[0]}: {c[1][:150]}\n    tpu: {t[0]}: {t[1][:250]}")
