"""LFW 다운로드 → data/lfw/{인물}/*.jpg 로 정리 (여러 경로 자동 시도)."""
import os
import shutil
import ssl
import tarfile
import urllib.request

OUT_ROOT = "data"
OUT_DIR = os.path.join(OUT_ROOT, "lfw")  # 최종: data/lfw/{인물}/*.jpg

MIRRORS = [
    "https://vis-www.cs.umass.edu/lfw/lfw.tgz",
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
    "https://ndownloader.figshare.com/files/5976018",
]

def _has_people():
    if not os.path.isdir(OUT_DIR):
        return False
    return any(os.path.isdir(os.path.join(OUT_DIR, d)) for d in os.listdir(OUT_DIR))

def _link_or_copy(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.exists(dst):
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)

def _via_torchvision():
    """torchvision 내장 다운로더 (미러·재시도·체크섬 포함)."""
    from torchvision.datasets import LFWPeople

    print("[1/3] torchvision으로 LFW 받는 중...")
    LFWPeople(root=OUT_ROOT, split="train", image_set="original", download=True)
    base = os.path.join(OUT_ROOT, "lfw-py")
    for cand in ("lfw_funneled", "lfw", "lfw-deepfunneled"):
        src = os.path.join(base, cand)
        if os.path.isdir(src):
            _link_or_copy(src, OUT_DIR)
            return True
    return False

def _via_sklearn():
    """sklearn fetch_lfw_people → 인물별 jpg 저장 (사진 많은 인물만)."""
    import numpy as np
    from PIL import Image
    from sklearn.datasets import fetch_lfw_people

    print("[2/3] scikit-learn으로 LFW 받는 중...")
    data = fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0)
    os.makedirs(OUT_DIR, exist_ok=True)
    counts = {}
    for img, target in zip(data.images, data.target):
        name = data.target_names[target].replace(" ", "_")
        d = os.path.join(OUT_DIR, name)
        os.makedirs(d, exist_ok=True)
        i = counts.get(name, 0)
        arr = (img * 255).astype("uint8") if img.max() <= 1.0 else img.astype("uint8")
        Image.fromarray(arr).save(os.path.join(d, f"{i:04d}.jpg"))
        counts[name] = i + 1
    print(f"  sklearn: 인물 {len(counts)}명 저장")
    return True

def _via_direct():
    """직접 tgz 다운로드 후 압축 해제 (미러 순회)."""
    print("[3/3] 직접 다운로드 시도...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    tgz = os.path.join(OUT_ROOT, "lfw.tgz")
    for url in MIRRORS:
        try:
            print(f"  시도: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r, \
                    open(tgz, "wb") as f:
                shutil.copyfileobj(r, f)
            with tarfile.open(tgz, "r:gz") as tar:
                tar.extractall(OUT_ROOT)
            return True
        except Exception as e:
            print(f"    실패: {e}")
    return False

def download_lfw():
    if _has_people():
        print(f"이미 준비됨 → {OUT_DIR}")
        return
    os.makedirs(OUT_ROOT, exist_ok=True)
    for fn in (_via_torchvision, _via_sklearn, _via_direct):
        try:
            if fn() and _has_people():
                print(f"\n완료 → {OUT_DIR}")
                print("다음 단계: python utils/prepare_lfw.py --lfw_dir data/lfw")
                return
        except Exception as e:
            print(f"  경로 실패: {e}")
    print("\n모든 자동 경로 실패. 인터넷/방화벽을 확인하거나, "
          "수동으로 lfw.tgz 를 받아 data/ 에 풀어 data/lfw/ 로 두세요.")

if __name__ == "__main__":
    download_lfw()
