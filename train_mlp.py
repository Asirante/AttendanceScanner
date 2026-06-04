"""exp06 — 고정된 VGGFace2 임베딩 위에 직접 설계한 MLP 분류기를 학습·평가."""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate import _metrics
from models.classifier import FaceClassifier
from utils.dataset import FaceDataset
from utils.registry import load_label_map

TRAIN_DIR = "data/splits/train"
VAL_DIR = "data/splits/val"
TEST_DIR = "data/splits/test"


class EmbeddingMLP(nn.Module):
    """512차원 임베딩 -> 은닉층 2개 -> 클래스. BatchNorm + Dropout 적용."""

    def __init__(self, num_classes: int, hidden=(512, 256), dropout=0.4):
        super().__init__()
        layers, dim = [], 512
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _extract_embeddings(backbone, root, lmap, device):
    """디렉토리의 모든 이미지를 임베딩(512d)으로 변환해 텐서로 반환."""
    ds = FaceDataset(root, lmap, augment=False)
    embs, labels = [], []
    with torch.no_grad():
        for img, label in DataLoader(ds, batch_size=64):
            e = F.normalize(backbone(img.to(device)), dim=1)
            embs.append(e.cpu())
            labels.append(label)
    return torch.cat(embs), torch.cat(labels)


def run_mlp(epochs=60, lr=1e-3, save_emb=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lmap = load_label_map()
    n = len(lmap)
    backbone = FaceClassifier(num_classes=n).backbone.eval().to(device)

    # 임베딩을 미리 한 번만 추출 (백본은 고정이므로 매 에폭 재계산 불필요 -> 빠름)
    print("임베딩 추출 중...")
    Xtr, ytr = _extract_embeddings(backbone, TRAIN_DIR, lmap, device)
    Xva, yva = _extract_embeddings(backbone, VAL_DIR, lmap, device)
    Xte, yte = _extract_embeddings(backbone, TEST_DIR, lmap, device)
    if save_emb:
        torch.save({"X": Xte, "y": yte, "lmap": lmap}, "runs/test_embeddings.pt")
        print("test 임베딩 저장: runs/test_embeddings.pt (t-SNE 시각화에 사용)")

    tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    model = EmbeddingMLP(n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    best_val, best_state, no_imp = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        # 검증
        model.eval()
        with torch.no_grad():
            va = (model(Xva.to(device)).argmax(1).cpu() == yva).float().mean().item()
        if va > best_val:
            best_val, best_state, no_imp = va, {k: v.cpu().clone()
                                                for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= 15:
                print(f"조기 종료 @ epoch {ep + 1}")
                break
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep + 1}: val_acc={va:.4f}")

    model.load_state_dict(best_state)
    print(f"best val_acc={best_val:.4f}")

    # 테스트 평가 (지표 + 속도)
    model.eval()
    all_preds, all_labels, latencies = [], [], []
    with torch.no_grad():
        for i in range(len(Xte)):
            x = Xte[i:i + 1].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            p = model(x).argmax(1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
            all_preds.append(p.item())
            all_labels.append(int(yte[i]))

    target_names = [lmap[i] for i in range(n)]
    res = _metrics(all_labels, all_preds, target_names, latencies)
    torch.save(best_state, "models/exp06_mlp_best.pth")
    print(f"\n[exp06_mlp] test_acc={res['test_acc']:.4f} "
          f"macro_f1={res['macro_f1']:.4f} fps={res['fps']:.1f}")
    return res


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    run_mlp()
