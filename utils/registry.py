"""직원 등록 & 임베딩 DB + label_map 유틸."""
import json
import os
import sqlite3

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from . import DB_PATH

def _get_backbone(device):
    return InceptionResnetV1(pretrained="vggface2").eval().to(device)

def _ensure_employees_table(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS Employees (
            emp_id    TEXT  PRIMARY KEY,
            name      TEXT  NOT NULL,
            embedding BLOB  NOT NULL
        );
        """
    )
    conn.commit()

def capture_embedding(
    emp_id: str,
    name: str,
    duration_sec: int = 5,
    n_frames: int = 10,
):
    """웹캠으로 얼굴 촬영 → 평균 임베딩(FP32) 계산 → Employees 테이블에 저장."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=20, post_process=False, device=device)
    backbone = _get_backbone(device)

    cap = cv2.VideoCapture(0)
    embeddings, collected = [], 0
    start = cv2.getTickCount()
    print(f"[{emp_id}] 촬영 시작 — {duration_sec}초 동안 정면을 바라보세요.")

    while cap.isOpened() and collected < n_frames:
        elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
        if elapsed > duration_sec:
            break
        ret, frame = cap.read()
        if not ret:
            break
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, probs = mtcnn.detect(img_pil)
        if probs is not None and probs[0] is not None and probs[0] >= 0.95:
            face = mtcnn(img_pil)
            if face is not None:
                face = ((face - 127.5) / 128.0).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = backbone(face)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                embeddings.append(emb.squeeze(0).cpu())
                collected += 1
                print(f"  {collected}/{n_frames} 수집", end="\r")
        cv2.imshow("직원 등록 — q로 중단", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not embeddings:
        raise RuntimeError("임베딩 수집 실패 — 얼굴이 감지되지 않았습니다.")

    mean_emb = torch.stack(embeddings).mean(dim=0)
    mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=0)
    emb_bytes = mean_emb.numpy().astype(np.float32).tobytes()

    conn = sqlite3.connect(DB_PATH)
    _ensure_employees_table(conn)
    conn.execute(
        "INSERT OR REPLACE INTO Employees (emp_id, name, embedding) VALUES (?, ?, ?)",
        (emp_id, name, emb_bytes),
    )
    conn.commit()
    conn.close()
    print(f"\n[완료] {name}({emp_id}) 등록됨 — {collected}프레임 평균 임베딩 저장")

def load_all_embeddings() -> dict:
    """DB에서 전체 직원 임베딩 로드."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT emp_id, name, embedding FROM Employees").fetchall()
    conn.close()
    result = {}
    for emp_id, name, emb_blob in rows:
        emb = torch.from_numpy(np.frombuffer(emb_blob, dtype=np.float32).copy())
        result[emp_id] = {"name": name, "embedding": emb}
    return result

def find_best_match(
    query_emb: torch.Tensor,
    db_embeddings: dict,
    threshold: float = 0.70,
) -> tuple:
    """query_emb: 실시간 추론에서 얻은 (512,) L2-정규화 FP32 벡터."""
    best_id, best_sim = None, -1.0
    for emp_id, data in db_embeddings.items():

        sim = torch.dot(query_emb, data["embedding"]).item()
        if sim > best_sim:
            best_sim, best_id = sim, emp_id
    if best_id is not None and best_sim >= threshold:
        return best_id, db_embeddings[best_id]["name"], best_sim
    return None, "unknown", best_sim

def build_label_map(splits_train_dir: str = "data/splits/train",
                    out_path: str = "data/label_map.json") -> dict:
    """train 디렉토리의 emp_id 폴더를 정렬하여 {idx: emp_id} 매핑 생성/저장."""
    emp_ids = sorted(
        d for d in os.listdir(splits_train_dir)
        if os.path.isdir(os.path.join(splits_train_dir, d))
    )
    label_map = {idx: emp_id for idx, emp_id in enumerate(emp_ids)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"label_map 저장: {out_path} ({len(label_map)}명)")
    return label_map

def load_label_map(path: str = "data/label_map.json") -> dict:
    """{int_idx: emp_id} 형태로 로드 (JSON 키는 문자열이므로 int 캐스팅)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

if __name__ == "__main__":

    capture_embedding("E001", "홍길동")
