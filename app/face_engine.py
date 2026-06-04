"""얼굴 임베딩 엔진 — MTCNN 검출 + InceptionResnetV1 임베딩."""
import numpy as np

class FaceEngine:
    def __init__(self, use_fp16: bool = True):
        self._loaded = False
        self.use_fp16 = use_fp16
        self.device = None
        self.mtcnn = None
        self.backbone = None
        self.torch = None

    def load(self, progress_cb=None):
        """모델 로드(최초 1회). progress_cb(str) 로 진행 상황 전달 가능."""
        if self._loaded:
            return
        if progress_cb:
            progress_cb("모델 로딩 중... (최초 실행은 가중치 다운로드로 시간이 걸립니다)")
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=160, margin=20, post_process=False, device=self.device
        )
        self.backbone = InceptionResnetV1(pretrained="vggface2").eval().to(
            self.device
        )
        self._fp16_active = self.use_fp16 and self.device.type == "cuda"
        if self._fp16_active:
            self.backbone.half()
        self._loaded = True
        if progress_cb:
            dev = "GPU" if self.device.type == "cuda" else "CPU"
            progress_cb(f"모델 로딩 완료 ({dev})")

    def embed_face(self, frame_bgr) -> np.ndarray:
        """BGR 프레임 1장 → L2 정규화된 512-dim 임베딩(float32) 또는 None."""
        self.load()
        import cv2
        from PIL import Image

        img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes, probs = self.mtcnn.detect(img_pil)
        if probs is None or probs[0] is None or probs[0] < 0.95:
            return None
        face = self.mtcnn(img_pil)
        if face is None:
            return None
        torch = self.torch
        inp = ((face - 127.5) / 128.0).unsqueeze(0).to(self.device)
        if getattr(self, "_fp16_active", False):
            inp = inp.half()
        with torch.no_grad():
            emb = self.backbone(inp)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb = emb.squeeze(0).float().cpu().numpy()
        return emb.astype(np.float32)

    def detect_box(self, frame_bgr):
        """프레임에서 첫 얼굴 박스 (x1,y1,x2,y2) 또는 None — 미리보기용."""
        self.load()
        import cv2
        from PIL import Image

        img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes, probs = self.mtcnn.detect(img_pil)
        if boxes is None or len(boxes) == 0:
            return None
        return tuple(int(v) for v in boxes[0])

    def cleanup(self):
        """장시간 운영 시 누적되는 GPU 캐시를 주기적으로 비운다."""
        if self._loaded and self.torch is not None:
            try:
                if self.device is not None and self.device.type == "cuda":
                    self.torch.cuda.empty_cache()
            except Exception:
                pass

def average_embeddings(embs: list) -> bytes:
    """여러 임베딩을 평균 → L2 정규화 → float32 BLOB 직렬화."""
    arr = np.stack(embs).mean(axis=0)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.astype(np.float32).tobytes()

def cosine_best_match(query: np.ndarray, db_embeddings: dict, threshold: float):
    """query(512,) vs DB → (emp_id, name, sim) | (None, 'unknown', best_sim)."""
    best_id, best_sim = None, -1.0
    for emp_id, data in db_embeddings.items():
        sim = float(np.dot(query, data["embedding"]))
        if sim > best_sim:
            best_sim, best_id = sim, emp_id
    if best_id is not None and best_sim >= threshold:
        return best_id, db_embeddings[best_id]["name"], best_sim
    return None, "unknown", best_sim

def open_camera_safe(index: int = 0, timeout: float = 3.0):
    """카메라를 열고 테스트 프레임을 timeout 안에 읽어본다.

    가상 카메라 등이 'isOpened는 통과하지만 read에서 멈추는' 경우, 별도 스레드에서
    열고 읽어 timeout을 적용한다. 성공하면 열린 VideoCapture를 반환, 실패하면 None.
    GUI 메인 스레드가 멈추는 것을 방지한다.
    """
    import threading

    import cv2

    result = {"cap": None, "ok": False}

    def _worker():
        cap = None
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") \
                else cv2.VideoCapture(index)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                return
            ok, frame = cap.read()
            if ok and frame is not None:
                result["cap"] = cap
                result["ok"] = True
            else:
                cap.release()
        except Exception:
            # cv2가 내부에서 C++ 예외를 던지는 경우(가상캠 등) 안전하게 무시
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or not result["ok"]:
        # timeout(멈춤) 또는 읽기 실패 — 사용 불가 카메라로 간주
        return None
    return result["cap"]

def list_cameras(max_index: int = 5) -> list:
    """사용 가능한 카메라 인덱스 목록을 탐색해 반환. 멈추는 카메라는 timeout으로 건너뜀."""
    found = []
    for i in range(max_index):
        cap = open_camera_safe(i, timeout=2.0)
        if cap is not None:
            found.append(i)
            cap.release()
    return found

def test_camera(index: int = 0, timeout: float = 3.0):
    """지정한 카메라를 timeout 안에 열어 한 프레임을 읽어 (성공여부, 해상도) 반환.

    가상캠 등이 멈추거나 C++ 예외를 던져도 안전하게 (False, None) 을 반환한다.
    """
    cap = open_camera_safe(index, timeout=timeout)
    if cap is None:
        return False, None
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, None
        h, w = frame.shape[:2]
        return True, (w, h)
    except Exception:
        return False, None
    finally:
        try:
            cap.release()
        except Exception:
            pass
