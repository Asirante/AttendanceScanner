"""성능 비교 실험 러너 — 여러 설정을 학습·평가하고 비교표(CSV)를 생성.

이미 완료된(=CSV에 기록된) 실험은 건너뛰므로, 중간에 멈춰도 다시 실행하면 이어서 진행된다.
"""
import csv
import os

from evaluate import run_embedding_baseline, run_evaluation
from train import train

# exp00: 학습 없이 VGGFace2 임베딩+유사도로만 평가 (실제 앱 방식, 베이스라인)
BASELINE = {"exp_name": "exp00_vggface2_embedding"}

# 학습 기반 비교 실험 — epochs 상향 + patience(조기 종료 여유) 지정
EXPERIMENTS = [
    {"exp_name": "exp01_pretrained_frozen",   # 전이학습: backbone 동결
     "freeze": True,  "unfreeze_blocks": None, "augment": True,  "lr": 1e-3, "epochs": 100, "patience": 15},
    {"exp_name": "exp02_finetune_last2",      # 전이학습: 마지막 2블록 미세조정
     "freeze": True,  "unfreeze_blocks": 2,    "augment": True,  "lr": 5e-4, "epochs": 100, "patience": 15},
    {"exp_name": "exp03_from_scratch",        # 처음부터 학습 (pretrained 미사용 비교군)
     "freeze": False, "unfreeze_blocks": None, "augment": True,  "lr": 1e-3, "epochs": 150, "patience": 20,
     "pretrained": False},
    {"exp_name": "exp04_no_augment",          # 증강 효과 확인 (동결 + 증강 off)
     "freeze": True,  "unfreeze_blocks": None, "augment": False, "lr": 1e-3, "epochs": 100, "patience": 15},
    {"exp_name": "exp05_lr_low",              # 학습률 비교
     "freeze": True,  "unfreeze_blocks": None, "augment": True,  "lr": 1e-4, "epochs": 100, "patience": 15},
]

TRAIN_DIR = "data/splits/train"
VAL_DIR = "data/splits/val"
TEST_DIR = "data/splits/test"
RESULT_CSV = "runs/comparison.csv"
FIELDS = ["experiment", "pretrained", "freeze", "unfreeze_blocks", "augment",
          "lr", "val_acc", "test_acc", "macro_f1", "ms_per_frame", "fps"]

def _load_done():
    """CSV에 이미 기록된 실험 이름 집합과 기존 행들을 반환."""
    if not os.path.exists(RESULT_CSV):
        return set(), []
    try:
        with open(RESULT_CSV, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        return {r["experiment"] for r in rows}, rows
    except Exception:
        return set(), []

def _append_row(row, write_header):
    """결과 한 줄을 CSV에 즉시 append (실험 끝날 때마다 저장 → 중단에 안전)."""
    os.makedirs("runs", exist_ok=True)
    mode = "w" if write_header else "a"
    with open(RESULT_CSV, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def _row_from(name, ev, pretrained="-", freeze="-", unfreeze="-",
              augment="-", lr="-", val_acc=""):
    return {
        "experiment": name, "pretrained": pretrained, "freeze": freeze,
        "unfreeze_blocks": unfreeze, "augment": augment, "lr": lr,
        "val_acc": val_acc,
        "test_acc": round(ev["test_acc"], 4),
        "macro_f1": round(ev["macro_f1"], 4),
        "ms_per_frame": round(ev["ms_per_frame"], 2),
        "fps": round(ev["fps"], 1),
    }

def run_all():
    """완료되지 않은 실험만 학습+평가하고, 끝날 때마다 CSV에 기록."""
    os.makedirs("runs", exist_ok=True)
    done, existing = _load_done()
    if done:
        print(f"이미 완료된 실험 건너뜀: {sorted(done)}")
    first_write = not existing

    # exp00 — 학습 없이 임베딩 베이스라인
    if BASELINE["exp_name"] not in done:
        print(f"\n===== {BASELINE['exp_name']} 평가 (학습 없음) =====")
        ev = run_embedding_baseline(TRAIN_DIR, TEST_DIR)
        _append_row(_row_from(BASELINE["exp_name"], ev), write_header=first_write)
        first_write = False
        print(f"  → 저장됨: {RESULT_CSV}")

    # exp01~05 — 학습 기반
    for cfg in EXPERIMENTS:
        if cfg["exp_name"] in done:
            continue
        cfg = {**cfg, "train_dir": TRAIN_DIR, "val_dir": VAL_DIR}
        print(f"\n===== {cfg['exp_name']} 학습 시작 =====")
        tr = train(cfg)
        print(f"===== {cfg['exp_name']} 평가 =====")
        ev = run_evaluation(tr["model_path"], TEST_DIR,
                            pretrained=cfg.get("pretrained", True))
        row = _row_from(
            cfg["exp_name"], ev,
            pretrained=cfg.get("pretrained", True), freeze=cfg["freeze"],
            unfreeze=cfg.get("unfreeze_blocks"), augment=cfg["augment"],
            lr=cfg["lr"], val_acc=round(tr["best_val_acc"], 4),
        )
        _append_row(row, write_header=first_write)
        first_write = False
        print(f"  → 저장됨: {RESULT_CSV}")

    _print_summary()

def _print_summary():
    _, rows = _load_done()
    if not rows:
        return
    print("\n=== 실험 비교 요약 ===")
    print(f"{'실험':28} {'정확도':>8} {'F1':>8} {'FPS':>8}")
    for r in rows:
        print(f"{r['experiment']:28} {r['test_acc']:>8} {r['macro_f1']:>8} {r['fps']:>8}")

if __name__ == "__main__":
    run_all()
