"""Adds the cRT (re-initialised) Stage 2 checkpoints and comparisons to src/evaluate_test.py.
Run from the project root after ALL reinit jobs have finished:  python scripts/add_reinit_to_evaluate_test.py"""
import shutil

PATH = "src/evaluate_test.py"
s = open(PATH).read()
if "c2_s2r" in s:
    print("Already added, nothing to do."); raise SystemExit
ck_anchor = '"config5_msc_stage2/checkpoint_best.pt"),\n'
cmp_anchor = '("c5_msc_s1", "c4_ral_s1"),     # contrastive vs RAL, Stage 1\n'
assert ck_anchor in s and cmp_anchor in s, "Config 5 lines not found; add the entries by hand."
new_ck = [("c2_s2r", "Config 2 BCE S2 cRT", "stage2_bce_reinit"),
          ("c3_wd2e4_s2r", "Config 3 WD 2e-4 S2 cRT", "config3_wd4_stage2_reinit"),
          ("c3_wd5e4_s2r", "Config 3 WD 5e-4 S2 cRT", "config3_wd5_stage2_reinit"),
          ("c3_wd1e3_s2r", "Config 3 WD 1e-3 S2 cRT", "config3_wd1_stage2_reinit"),
          ("c3_wd5e3_s2r", "Config 3 WD 5e-3 S2 cRT", "config3_wd2_stage2_reinit"),
          ("c3_wd1e2_s2r", "Config 3 WD 1e-2 S2 cRT", "config3_wd3_stage2_reinit"),
          ("c4_ral_s2r", "Config 4 RAL S2 cRT", "config4_ral_stage2_reinit"),
          ("c5_msc_s2r", "Config 5 MSC S2 cRT", "config5_msc_stage2_reinit")]
ck_lines = "".join(f'    "{k}": ("{d}", "{p}/checkpoint_best.pt"),\n' for k, d, p in new_ck)
pairs = [("c2_s2r", "c2_s1"), ("c3_wd2e4_s2r", "c3_wd2e4_s1"), ("c4_ral_s2r", "c4_ral_s1"), ("c5_msc_s2r", "c5_msc_s1"),
         ("c2_s2r", "c2_s2"), ("c3_wd2e4_s2r", "c3_wd2e4_s2"), ("c4_ral_s2r", "c4_ral_s2"), ("c5_msc_s2r", "c5_msc_s2"),
         ("c3_wd2e4_s2r", "c2_s2r"), ("c4_ral_s2r", "c2_s2r"), ("c5_msc_s2r", "c2_s2r"), ("c2_s2r", "c1_joint")]
cmp_lines = "".join(f'    ("{a}", "{b}"),\n' for a, b in pairs)
shutil.copy(PATH, PATH + ".bak2")
s = s.replace(ck_anchor, ck_anchor + ck_lines, 1).replace(cmp_anchor, cmp_anchor + cmp_lines, 1)
open(PATH, "w").write(s)
print(f"Added {len(new_ck)} checkpoints and {len(pairs)} comparisons (backup: src/evaluate_test.py.bak2)")
