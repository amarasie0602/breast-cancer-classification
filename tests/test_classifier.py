import torch

from src.models.classifier import BreakHisClassifier


def test_forward_pass_output_shape():
    model = BreakHisClassifier(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1)


def test_freeze_backbone_keeps_fc_trainable():
    model = BreakHisClassifier(pretrained=False)
    model.freeze_backbone()

    assert all(p.requires_grad for p in model.backbone.fc.parameters())
    non_fc_params = [
        p for name, p in model.backbone.named_parameters() if not name.startswith("fc.")
    ]
    assert not any(p.requires_grad for p in non_fc_params)


def test_unfreeze_backbone_makes_all_trainable():
    model = BreakHisClassifier(pretrained=False)
    model.freeze_backbone()
    model.unfreeze_backbone()

    assert all(p.requires_grad for p in model.backbone.parameters())
