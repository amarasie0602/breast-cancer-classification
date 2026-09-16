import torch

from src.explainability.gradcam import GradCAM
from src.models.classifier import BreakHisClassifier


def test_gradcam_output_shape_and_range():
    model = BreakHisClassifier(pretrained=False)
    model.eval()

    target_layer = model.backbone.layer4[-1]
    cam_extractor = GradCAM(model, target_layer)

    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    cam = cam_extractor(x)

    assert cam.shape[0] == 2
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-6
