"""exp07 — ArcFace(각도 마진) 분류기를 임베딩 위에 학습. margin/scale 그리드 비교."""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate import _metrics
from train_mlp import _extract_embeddings  # 임베딩 추출 재사용
from models.classifier import FaceClassifier
from utils.registry import load_label_map

TRAIN_DIR = "data/splits/train"
VAL_DIR = "data/splits/val"
TEST_DIR = "data/splits/test"

# margin(m) x scale(s) 그리드 — 4 x 3 = 12 조합 비교
# margin: 작을수록 약한 마진, 클수록 강한 분리 압력
# scale : 로짓 스케일. 너무 작으면 학습이 약하고, 크면 과해질 수 있음
MARGINS = [0.2, 0.35, 0.5, 0.7]
SCALES = [16.0, 32.0, 64.0]
GRID = [{"m": m, "s": s} for m in MARGINS for s in SCALES]
EPOCHS = 80


class ArcFaceHead(nn.Module):
    """정규화된 임베딩 -> 정규화된 클래스 가중치와의 각도에 마진을 더해 분류.

    s = 스케일, m = 각도 마진(라디안). 학습 시에만 마진 적용, 추론은 일반 cos 유사도.
    """

    def __init__(self, in_dim, num_classes, s=32.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_normal_(self.W)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)        # 수치 안정용 임계
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        x = F.normalize(x, dim=1)
        W = F.normalize(self.W, dim=1)
        cos = F.linear(x, W).clamp(-1 + 1e-7, 1 - 1e-7)
        if label is None:                       # 추론: 마진 없이 점수만
            return self.s * cos
        sin = torch.sqrt(1.0 - cos ** 2)
        phi = cos * self.cos_m - sin * self.sin_m   # cos(θ + m)
        phi = torch.where(cos > self.th, phi, cos - self.mm)  # 안정화
        onehot = F.one_hot(label, cos.size(1)).float()
        out = onehot * phi + (1 - onehot) * cos     # 정답 클래스에만 마진
        return self.s * out


def _run_one(cfg, data, device, n, lmap):
    Xtr, ytr, Xva, yva, Xte, yte = data
    name = f"exp07_arcface_m{cfg['m']}_s{int(cfg['s'])}"
    head = ArcFaceHead(512, n, s=cfg["s"], m=cfg["m"]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

    best_val, best_state, no_imp = 0.0, None, 0
    for ep in range(EPOCHS):
        head.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(head(xb, yb), yb)
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            va = (head(Xva.to(device)).argmax(1).cpu() == yva).float().mean().item()
        if va > best_val:
            best_val, no_imp = va, 0
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= 15:
                break
    head.load_state_dict(best_state)

    # test 평가
    head.eval()
    preds, labels, lat = [], [], []
    with torch.no_grad():
        for i in range(len(Xte)):
            x = Xte[i:i + 1].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            p = head(x).argmax(1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
            preds.append(p.item())
            labels.append(int(yte[i]))
    target = [lmap[i] for i in range(n)]
    res = _metrics(labels, preds, target, lat)
    print(f"[{name}] val={best_val:.4f} test_acc={res['test_acc']:.4f} "
          f"macro_f1={res['macro_f1']:.4f}")
    return name, cfg, best_val, res


def run_arcface():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lmap = load_label_map()
    n = len(lmap)
    backbone = FaceClassifier(num_classes=n).backbone.eval().to(device)
    print("임베딩 추출 중...")
    data = (
        *_extract_embeddings(backbone, TRAIN_DIR, lmap, device),
        *_extract_embeddings(backbone, VAL_DIR, lmap, device),
        *_extract_embeddings(backbone, TEST_DIR, lmap, device),
    )

    rows = []
    for cfg in GRID:
        print(f"\n===== ArcFace m={cfg['m']} s={cfg['s']} 학습 =====")
        name, c, val, res = _run_one(cfg, data, device, n, lmap)
        rows.append((name, c["m"], c["s"], val,
                     res["test_acc"], res["macro_f1"], res["fps"]))

    # 요약 출력
    print("\n=== ArcFace 그리드 요약 ===")
    print(f"{'margin':>7}{'scale':>7}{'val':>9}{'test_acc':>10}{'macro_f1':>10}")
    best = None
    for name, m, s, val, acc, f1, fps in rows:
        print(f"{m:>7}{s:>7}{val:>9.4f}{acc:>10.4f}{f1:>10.4f}")
        if best is None or acc > best[4]:
            best = (name, m, s, val, acc, f1, fps)
    print(f"\n최고: margin={best[1]} scale={best[2]} "
          f"test_acc={best[4]:.4f} macro_f1={best[5]:.4f}")

    # csv 저장
    import csv
    import os
    os.makedirs("runs", exist_ok=True)
    with open("runs/arcface_grid.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "margin", "scale", "val_acc",
                    "test_acc", "macro_f1", "fps"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], round(r[3], 4),
                        round(r[4], 4), round(r[5], 4), round(r[6], 1)])
    print("저장: runs/arcface_grid.csv")
    return rows


if __name__ == "__main__":
    run_arcface()
