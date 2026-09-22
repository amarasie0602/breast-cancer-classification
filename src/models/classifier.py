"""ResNet50-based binary classifier for BreakHis (transfer learning)."""

from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50


def build_backbone(pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    return resnet50(weights=weights)


class BreakHisClassifier(nn.Module):
    """ResNet50 with its final fc layer replaced for binary classification."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = build_backbone(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
