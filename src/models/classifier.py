"""ResNet50-based binary classifier for BreakHis (transfer learning)."""

from torchvision.models import ResNet50_Weights, resnet50


def build_backbone(pretrained=True):
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    return resnet50(weights=weights)
