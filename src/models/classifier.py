"""ResNet50-based binary classifier for BreakHis (transfer learning)."""

from torch import Tensor, nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)


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


class MalignantSubtypeClassifier(nn.Module):
    """EfficientNet-B0 with its classifier head replaced for malignant-subtype
    classification (multi-class, CrossEntropyLoss over logits -- unlike
    BreakHisClassifier's single-logit binary output).

    EfficientNet-B0 rather than the binary model's ResNet50 on purpose: this
    task has only 4-6 training patients for 3 of its 4 classes, and a 5.3M
    parameter backbone overfits that far less readily than a 25.6M one. It's
    also ~4x cheaper per epoch on CPU, which buys more epochs in the same
    wall clock. The binary classifier keeps ResNet50 -- it has 82 patients
    to learn from and is already trained and validated.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if not name.startswith("classifier."):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
