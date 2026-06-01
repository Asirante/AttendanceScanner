"""동영상/웹캠 → MTCNN 필터링 → raw float 텐서(.pt) 저장."""
import os

import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image

class FaceDataCollector:
    def __init__(self, emp_id: str, save_dir: str, conf_thr: float = 0.95):
        self.emp_id = emp_id
        self.save_dir = save_dir
        self.conf_thr = conf_thr
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            post_process=False,
            device=device,
        )

    def _save_face(self, img_pil: Image.Image, out_dir: str, saved: int) -> bool:
        """검출/필터/저장 공통 로직. 저장 성공 시 True."""
        boxes, probs = self.mtcnn.detect(img_pil)
        if probs is not None and probs[0] is not None and probs[0] >= self.conf_thr:
            face_tensor = self.mtcnn(img_pil)
            if face_tensor is not None:
                path = os.path.join(out_dir, f"{saved:04d}.pt")
                torch.save(face_tensor, path)
                return True
        return False

    def collect_from_video(self, video_path: str, frame_skip: int = 5) -> int:
        """동영상 → MTCNN 필터링 → .pt 텐서 저장. 저장 프레임 수 반환."""
        cap = cv2.VideoCapture(video_path)
        out_dir = os.path.join(self.save_dir, self.emp_id)
        os.makedirs(out_dir, exist_ok=True)
        saved, frame_idx = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self._save_face(img_pil, out_dir, saved):
                saved += 1
            frame_idx += 1
        cap.release()
        return saved

    def collect_live(self, duration_sec: int = 8) -> int:
        """웹캠 라이브 캡처 버전 — 동영상 파일 없이 직접 수집."""
        cap = cv2.VideoCapture(0)
        out_dir = os.path.join(self.save_dir, self.emp_id)
        os.makedirs(out_dir, exist_ok=True)
        saved, frame_idx = 0, 0
        start = cv2.getTickCount()
        while cap.isOpened():
            elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
            if elapsed > duration_sec:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 5 == 0:
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if self._save_face(img_pil, out_dir, saved):
                    saved += 1
            frame_idx += 1
        cap.release()
        return saved

if __name__ == "__main__":

    collector = FaceDataCollector(emp_id="E001", save_dir="data/raw")
    n = collector.collect_from_video("data/videos/E001.mp4", frame_skip=5)
    print(f"E001: {n} 프레임 저장")
