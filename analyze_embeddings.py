"""임베딩 분석·시각화 — t-SNE, 혼동행렬 히트맵, 유사도 임계값/ROC 분석."""
import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # 창 없이 파일로 저장
import matplotlib.pyplot as plt

OUT = "runs/figs"


def _load_test_embeddings():
    d = torch.load("runs/test_embeddings.pt")
    return d["X"].numpy(), d["y"].numpy(), d["lmap"]


def tsne_plot(X, y, lmap, max_classes=15):
    """임베딩을 2D로 투영해 사람별로 잘 뭉치는지 시각화."""
    from sklearn.manifold import TSNE
    # 사람이 많으면 보기 어려우므로 표본 수 많은 상위 N명만
    uniq, cnt = np.unique(y, return_counts=True)
    top = uniq[np.argsort(-cnt)[:max_classes]]
    mask = np.isin(y, top)
    Xs, ys = X[mask], y[mask]
    emb2d = TSNE(n_components=2, perplexity=30, init="pca",
                 random_state=42).fit_transform(Xs)
    plt.figure(figsize=(10, 8))
    for c in top:
        m = ys == c
        plt.scatter(emb2d[m, 0], emb2d[m, 1], s=18,
                    label=lmap.get(int(c), str(c)))
    plt.title(f"t-SNE of face embeddings (top {max_classes} people)")
    plt.legend(fontsize=7, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(f"{OUT}/tsne.png", dpi=140)
    plt.close()
    print(f"저장: {OUT}/tsne.png")


def threshold_roc(X, y):
    """동일인/타인 쌍의 코사인 유사도 분포로 ROC·EER 계산 (검증 성격)."""
    from sklearn.metrics import roc_curve, auc
    Xt = torch.tensor(X)
    sim = (Xt @ Xt.t()).numpy()  # 코사인 유사도 행렬 (이미 정규화됨)
    n = len(y)
    iu = np.triu_indices(n, k=1)
    scores = sim[iu]
    same = (y[iu[0]] == y[iu[1]]).astype(int)  # 1=동일인, 0=타인
    fpr, tpr, thr = roc_curve(same, scores)
    roc_auc = auc(fpr, tpr)
    # EER (FAR=FRR 지점)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_thr = thr[eer_idx]

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.scatter([fpr[eer_idx]], [tpr[eer_idx]], color="red",
                label=f"EER={eer:.4f} @ thr={eer_thr:.3f}")
    plt.xlabel("False Accept Rate")
    plt.ylabel("True Accept Rate")
    plt.title("Verification ROC (cosine similarity)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/roc.png", dpi=140)
    plt.close()
    print(f"저장: {OUT}/roc.png  (AUC={roc_auc:.4f}, EER={eer:.4f}, 권장 임계값~{eer_thr:.3f})")
    return roc_auc, eer, eer_thr


def confusion_heatmap(X, y, lmap, max_classes=20):
    """프로토타입 최근접 분류의 혼동행렬을 히트맵으로 (상위 N명)."""
    from sklearn.metrics import confusion_matrix
    uniq, cnt = np.unique(y, return_counts=True)
    top = uniq[np.argsort(-cnt)[:max_classes]]
    mask = np.isin(y, top)
    Xs, ys = torch.tensor(X[mask]), y[mask]
    # 클래스별 프로토타입
    protos, labels = [], []
    for c in top:
        protos.append(F.normalize(torch.tensor(X[y == c]).mean(0, keepdim=True), dim=1))
        labels.append(c)
    P = torch.cat(protos)
    pred_idx = (Xs @ P.t()).argmax(1).numpy()
    preds = np.array([labels[i] for i in pred_idx])
    cm = confusion_matrix(ys, preds, labels=top)
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    names = [lmap.get(int(c), str(c)) for c in top]
    plt.xticks(range(len(top)), names, rotation=90, fontsize=6)
    plt.yticks(range(len(top)), names, fontsize=6)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion matrix (top {max_classes} people)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/confusion.png", dpi=140)
    plt.close()
    print(f"저장: {OUT}/confusion.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X, y, lmap = _load_test_embeddings()
    print(f"임베딩 로드: {X.shape[0]}개, {X.shape[1]}차원")
    tsne_plot(X, y, lmap)
    confusion_heatmap(X, y, lmap)
    threshold_roc(X, y)
    print("\n완료 — runs/figs/ 에 tsne.png, confusion.png, roc.png 생성")
