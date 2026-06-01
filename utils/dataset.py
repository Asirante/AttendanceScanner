"""PyTorch Dataset — raw float .pt 텐서를 로드하여 학습용 정규화 텐서 반환."""
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, root_dir: str, label_map: dict, augment: bool = False):
        """root_dir: data/splits/{train|val|test}/"""
        self.samples = []

        reverse_map = {v: k for k, v in label_map.items()}

        for emp_id in sorted(os.listdir(root_dir)):
            if emp_id not in reverse_map:
                continue
            idx = reverse_map[emp_id]
            emp_dir = os.path.join(root_dir, emp_id)
            if not os.path.isdir(emp_dir):
                continue
            for fname in os.listdir(emp_dir):
                if fname.endswith(".pt"):
                    self.samples.append((os.path.join(emp_dir, fname), idx))

        self.aug = (
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.1
                    ),
                ]
            )
            if augment
            else None
        )

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path)

        if self.aug is not None:

            tensor = (tensor / 255.0).clamp(0.0, 1.0)
            tensor = self.aug(tensor)

            tensor = tensor * 2.0 - 1.0
        else:

            tensor = (tensor - 127.5) / 128.0

        return tensor, label

    def __len__(self):
        return len(self.samples)
