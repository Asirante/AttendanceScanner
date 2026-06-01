"""PyInstaller 런타임 훅 — 번들에 포함된 모델 가중치를 torch 가 찾도록 경로 설정.

onedir 로 빌드되면 spec 의 datas 항목이 exe 와 같은 폴더(_MEIPASS 가 아니라
실행 폴더)에 풀린다. 여기서 TORCH_HOME 을 그 위치로 잡아주면, facenet-pytorch 가
가중치를 인터넷에서 새로 받지 않고 번들된 파일을 사용한다(오프라인 첫 실행 가능).
"""
import os
import sys


def _set_torch_home():
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    cache_root = os.path.join(base, ".cache", "torch")
    if os.path.isdir(os.path.join(cache_root, "checkpoints")):
        os.environ.setdefault("TORCH_HOME", cache_root)


_set_torch_home()
