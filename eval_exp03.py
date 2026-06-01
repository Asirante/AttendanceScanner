"""이미 학습된 exp03 체크포인트만 평가해 CSV에 추가 (재학습 없이 복구용)."""
import csv, os
from evaluate import run_evaluation
from run_experiments import RESULT_CSV, FIELDS, _append_row, _load_done

CKPT = "models/exp03_from_scratch_best.pth"

ev = run_evaluation(CKPT, "data/splits/test", pretrained=False)
row = {
    "experiment": "exp03_from_scratch", "pretrained": False, "freeze": False,
    "unfreeze_blocks": None, "augment": True, "lr": 1e-3, "val_acc": 0.7734,
    "test_acc": round(ev["test_acc"], 4), "macro_f1": round(ev["macro_f1"], 4),
    "ms_per_frame": round(ev["ms_per_frame"], 2), "fps": round(ev["fps"], 1),
}
done, existing = _load_done()
_append_row(row, write_header=not existing)
print("\nexp03 결과 CSV에 추가 완료")
