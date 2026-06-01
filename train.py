"""분류기 학습 루프 (EXP-1~6 실험용)."""
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.classifier import FaceClassifier
from utils.dataset import FaceDataset
from utils.registry import build_label_map, load_label_map

def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0

def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs/{config['exp_name']}")

    label_map_path = "data/label_map.json"
    if not os.path.exists(label_map_path):
        build_label_map(config["train_dir"], label_map_path)
    lmap = load_label_map(label_map_path)

    train_ds = FaceDataset(config["train_dir"], lmap,
                           augment=config.get("augment", True))
    val_ds = FaceDataset(config["val_dir"], lmap, augment=False)
    tl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = FaceClassifier(
        num_classes=len(lmap),
        freeze_backbone=config["freeze"],
        pretrained=config.get("pretrained", True),
    ).to(device)
    if config.get("unfreeze_blocks"):
        model.unfreeze_last_blocks(config["unfreeze_blocks"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    best_val, no_imp = 0.0, 0
    patience = config.get("patience", 15)
    for epoch in range(config["epochs"]):
        model.train()
        t_loss = 0.0
        for imgs, labels in tl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        val_acc = evaluate(model, vl, device)
        scheduler.step()

        writer.add_scalar("Loss/train", t_loss / len(tl), epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        print(
            f"Epoch {epoch + 1:3d} | loss={t_loss / len(tl):.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val, no_imp = val_acc, 0
            torch.save(model.state_dict(), f"models/{config['exp_name']}_best.pth")
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping @ epoch {epoch + 1}")
                break

    writer.close()
    print(f"Best val_acc: {best_val:.4f}")
    return {"best_val_acc": best_val,
            "model_path": f"models/{config['exp_name']}_best.pth"}

if __name__ == "__main__":
    train(
        {
            "exp_name": "exp01_transfer_frozen",
            "train_dir": "data/splits/train",
            "val_dir": "data/splits/val",
            "epochs": 50,
            "lr": 1e-3,
            "freeze": True,
            "unfreeze_blocks": None,
        }
    )
