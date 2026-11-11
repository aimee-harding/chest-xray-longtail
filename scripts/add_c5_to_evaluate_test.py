"""Adds Config 5 to src/evaluate_test.py (makes a .bak copy first; safe to run twice).
Run from the project root:  python scripts/add_c5_to_evaluate_test.py"""
import shutil

PATH = "src/evaluate_test.py"
s = open(PATH).read()
if "c5_msc_s1" in s:
    print("Config 5 is already in evaluate_test.py, nothing to do."); raise SystemExit

ck_anchor = '"config4_ral_stage2/checkpoint_best.pt"),\n'
cmp_anchor = '("c4_ral_s2", "c4_ral_s1"),     # effect of Stage 2 on the RAL backbone\n'
assert ck_anchor in s and cmp_anchor in s, "Could not find the Config 4 lines; add the Config 5 lines by hand."

s = s.replace(ck_anchor, ck_anchor +
    '    "c5_msc_s1":    ("Config 5 MSC S1",             "config5_msc_stage1/checkpoint_best.pt"),\n'
    '    "c5_msc_s2":    ("Config 5 MSC S2",             "config5_msc_stage2/checkpoint_best.pt"),\n', 1)
s = s.replace(cmp_anchor, cmp_anchor +
    '    ("c5_msc_s1", "c2_s1"),         # contrastive vs BCE representation, Stage 1\n'
    '    ("c5_msc_s2", "c2_s2"),         # contrastive vs BCE, after Stage 2\n'
    '    ("c5_msc_s2", "c5_msc_s1"),     # effect of Stage 2 on the contrastive backbone\n'
    '    ("c5_msc_s1", "c4_ral_s1"),     # contrastive vs RAL, Stage 1\n', 1)
shutil.copy(PATH, PATH + ".bak")
open(PATH, "w").write(s)
print("Added c5_msc_s1 / c5_msc_s2 and 4 comparisons (backup: src/evaluate_test.py.bak)")
