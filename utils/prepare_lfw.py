"""LFW 데이터셋 → MTCNN 정렬 → data/raw/{인물}/*.pt 변환 (학습 실험용)."""
import argparse
import os

import torch
from facenet_pytorch import MTCNN
from PIL import Image

# data_collector.py 와 동일한 MTCNN 설정 (post_process=False → 0~255 float 텐서)
def _make_mtcnn():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MTCNN(image_size=160, margin=20, keep_all=False,
                 post_process=False, device=device)

def prepare_lfw(lfw_dir: str, out_dir: str = "data/raw",
                min_images: int = 20, conf_thr: float = 0.95):
    """LFW 원본(인물별 폴더의 jpg) → 얼굴 정렬 .pt 저장. 인물 수/프레임 수 반환."""
    mtcnn = _make_mtcnn()
    people = sorted(
        d for d in os.listdir(lfw_dir)
        if os.path.isdir(os.path.join(lfw_dir, d))
    )
    kept_people, total = 0, 0
    for person in people:
        src = os.path.join(lfw_dir, person)
        imgs = [f for f in os.listdir(src)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        # 이미지가 너무 적은 인물은 분류 학습에 부적합하므로 제외
        if len(imgs) < min_images:
            continue
        dst = os.path.join(out_dir, person)
        os.makedirs(dst, exist_ok=True)
        saved = 0
        for fname in imgs:
            try:
                img = Image.open(os.path.join(src, fname)).convert("RGB")
            except Exception:
                continue
            boxes, probs = mtcnn.detect(img)
            if probs is None or probs[0] is None or probs[0] < conf_thr:
                continue
            face = mtcnn(img)
            if face is None:
                continue
            torch.save(face, os.path.join(dst, f"{saved:04d}.pt"))
            saved += 1
        if saved >= min_images:
            kept_people += 1
            total += saved
            print(f"  {person}: {saved}장")
        else:
            # 정렬 후 부족해진 경우 폴더 정리
            for f in os.listdir(dst):
                os.remove(os.path.join(dst, f))
            os.rmdir(dst)
    print(f"\n완료 — 인물 {kept_people}명, 총 {total}장 → {out_dir}")
    print("다음 단계: python utils/split_dataset.py")
    return kept_people, total

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw_dir", default="data/lfw",
                    help="LFW 압축 해제 경로 (인물별 하위 폴더 포함)")
    ap.add_argument("--out_dir", default="data/raw")
    ap.add_argument("--min_images", type=int, default=20,
                    help="이 장수 미만인 인물은 제외 (기본 20)")
    args = ap.parse_args()
    prepare_lfw(args.lfw_dir, args.out_dir, args.min_images)
