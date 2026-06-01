"""Test set 최종 평가 + Inference Time 측정."""
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from models.classifier import FaceClassifier
from utils.dataset import FaceDataset
from utils.registry import load_label_map

def _metrics(all_labels, all_preds, target_names, latencies):
    """공통: 정확도/F1/혼동행렬 출력 + 지표 dict 반환."""
    report = classification_report(
        all_labels, all_preds, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    print(classification_report(all_labels, all_preds,
                                target_names=target_names, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    avg_ms = sum(latencies) / len(latencies)
    fps = 1000 / avg_ms
    print(f"\nInference Time: {avg_ms:.2f} ms/frame  ({fps:.1f} FPS)")
    print(f"목표 FPS >=15: {'달성' if fps >= 15 else '미달 — 최적화 필요'}")
    return {
        "test_acc": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "ms_per_frame": avg_ms,
        "fps": fps,
    }

def run_embedding_baseline(train_dir: str, test_dir: str):
    """학습 없이 VGGFace2 임베딩 + 코사인 유사도로 분류 (실제 앱 방식, exp00)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lmap = load_label_map()
    n = len(lmap)
    backbone = FaceClassifier(num_classes=n).backbone.eval().to(device)

    # 1) train 으로 클래스별 평균 임베딩(프로토타입) 구성
    train_ds = FaceDataset(train_dir, lmap, augment=False)
    sums = torch.zeros(n, 512, device=device)
    counts = torch.zeros(n, device=device)
    with torch.no_grad():
        for img, label in DataLoader(train_ds, batch_size=64):
            emb = F.normalize(backbone(img.to(device)), dim=1)
            for e, lb in zip(emb, label):
                sums[lb] += e
                counts[lb] += 1
    protos = F.normalize(sums / counts.clamp(min=1).unsqueeze(1), dim=1)

    # 2) test 를 가장 가까운 프로토타입으로 분류
    test_ds = FaceDataset(test_dir, lmap, augment=False)
    all_preds, all_labels, latencies = [], [], []
    with torch.no_grad():
        for img, label in DataLoader(test_ds, batch_size=1):
            img = img.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            emb = F.normalize(backbone(img), dim=1)
            pred = (emb @ protos.t()).argmax(1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
            all_preds.append(pred.item())
            all_labels.append(label.item())

    target_names = [lmap[i] for i in range(n)]
    return _metrics(all_labels, all_preds, target_names, latencies)

def run_evaluation(model_path: str, test_dir: str, pretrained: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lmap = load_label_map()
    test_ds = FaceDataset(test_dir, lmap, augment=False)

    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = FaceClassifier(num_classes=len(lmap), pretrained=pretrained).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels, latencies = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds = model(imgs).argmax(1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    target_names = [lmap[i] for i in range(len(lmap))]
    return _metrics(all_labels, all_preds, target_names, latencies)

if __name__ == "__main__":
    run_evaluation("models/exp01_pretrained_frozen_best.pth", "data/splits/test")
