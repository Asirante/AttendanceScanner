"""raw_dir/{emp_id}/*.pt → out_dir/{train|val|test}/{emp_id}/*.pt  (70/15/15 분할)."""
import glob
import os
import shutil

from sklearn.model_selection import train_test_split

MIN_FRAMES = 10

def split_dataset(
    raw_dir: str,
    out_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    min_frames: int = MIN_FRAMES,
):
    for emp_id in sorted(os.listdir(raw_dir)):
        emp_path = os.path.join(raw_dir, emp_id)
        if not os.path.isdir(emp_path):
            continue
        frames = glob.glob(os.path.join(emp_path, "*.pt"))

        if len(frames) < min_frames:
            print(f"[경고] {emp_id}: 프레임 {len(frames)}개 — 추가 수집 권장 (skip)")
            continue

        train_f, tmp = train_test_split(
            frames, test_size=1 - train_ratio, random_state=42
        )

        val_f, test_f = train_test_split(tmp, test_size=0.5, random_state=42)

        for split, files in [("train", train_f), ("val", val_f), ("test", test_f)]:
            dest = os.path.join(out_dir, split, emp_id)
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest)
        print(f"  {emp_id}: train={len(train_f)} / val={len(val_f)} / test={len(test_f)}")

if __name__ == "__main__":
    split_dataset(raw_dir="data/raw", out_dir="data/splits")
