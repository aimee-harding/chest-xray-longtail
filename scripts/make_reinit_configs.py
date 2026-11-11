"""Makes a *_reinit copy of every Stage 2 config: same settings, new output folder, reinit_classifier: true."""
import re

SOURCES = ["stage2_bce", "config3_wd4_stage2", "config3_wd5_stage2", "config3_wd1_stage2",
           "config3_wd2_stage2", "config3_wd3_stage2", "config4_ral_stage2", "config5_msc_stage2"]
for name in SOURCES:
    s = open(f"configs/{name}.yaml").read()
    assert s.count("\nmodel:\n") == 1, f"{name}: no 'model:' section"
    s = re.sub(r'(\n  name: "[^"]+)"', r'\1_reinit"', s, count=1)
    s = re.sub(r'(\n  save_dir: "[^"]+)"', r'\1_reinit"', s, count=1)
    s = s.replace("\nmodel:\n", "\nmodel:\n  reinit_classifier: true\n", 1)
    open(f"configs/{name}_reinit.yaml", "w").write(s)
    out = re.search(r'save_dir: "([^"]+)"', s).group(1)
    print("configs/" + name + "_reinit.yaml  ->  " + out)
