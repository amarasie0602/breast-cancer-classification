"""ResNet50-based binary classifier for BreakHis (transfer learning)."""

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_backbone(pretrained=True):
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    return resnet50(weights=weights)


class BreakHisClassifier(nn.Module):
    """ResNet50 with its final fc layer replaced for binary classification."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = build_backbone(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)
