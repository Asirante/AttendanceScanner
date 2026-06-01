"""학습용 분류기 모델."""
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FaceClassifier(nn.Module):
    def __init__(self, num_classes: int = 30, freeze_backbone: bool = True,
                 pretrained: bool = True):
        super().__init__()

        self.backbone = InceptionResnetV1(
            pretrained="vggface2" if pretrained else None,
            classify=False,
            num_classes=None,
        )
        self.classifier = nn.Linear(512, num_classes)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(x)
        return self.classifier(embedding)

    def unfreeze_last_blocks(self, n: int = 2):
        """EXP-2: 마지막 n개 block 만 Fine-tuning 해동."""
        blocks = list(self.backbone.named_children())
        for name, module in blocks[-n:]:
            for p in module.parameters():
                p.requires_grad = True
            print(f"Unfrozen block: {name}")

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """분류 없이 512-dim 임베딩만 추출."""
        with torch.no_grad():
            return self.backbone(x)
