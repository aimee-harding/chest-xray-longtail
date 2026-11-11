"""Adds an optional 'reinit_classifier' flag to src/stage2_trainer.py (Kang et al. cRT).
With the flag off (all existing configs), behaviour is unchanged.
Run from the project root:  python scripts/patch_stage2_reinit.py   (makes a .bak copy; safe to run twice)"""
import shutil

PATH = "src/stage2_trainer.py"
s = open(PATH).read()
if "reinit_classifier" in s:
    print("stage2_trainer.py already has reinit_classifier, nothing to do."); raise SystemExit

anchor = '        self.model.load_state_dict(checkpoint["model_state_dict"])\n'
assert s.count(anchor) == 1, "Could not find the load_state_dict line; add the block by hand (see README)."
block = (
    '\n'
    '        # cRT (Kang et al., 2020): discard the Stage 1 classifier and learn a new one from scratch.\n'
    '        # Same initialisation as a new DenseNet121Backbone head.\n'
    '        if config["model"].get("reinit_classifier", False):\n'
    '            torch.nn.init.xavier_uniform_(self.model.classifier.weight)\n'
    '            torch.nn.init.zeros_(self.model.classifier.bias)\n'
    '            print("[Stage2Trainer] Classifier re-initialised (cRT)")\n'
    '        else:\n'
    '            print("[Stage2Trainer] Classifier initialised from Stage 1")\n'
)
shutil.copy(PATH, PATH + ".bak")
open(PATH, "w").write(s.replace(anchor, anchor + block, 1))
print("Patched src/stage2_trainer.py (backup: src/stage2_trainer.py.bak)")
