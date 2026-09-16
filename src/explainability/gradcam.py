"""Grad-CAM for visualizing which regions of a histology image drove a prediction."""

import torch
import torch.nn.functional as F


class GradCAM:
    """Registers forward/backward hooks on a target conv layer to compute Grad-CAM."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if target_class is None:
            score = logits.squeeze(1) if logits.dim() > 1 else logits
        else:
            score = logits[:, target_class]
        score.sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))

        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = cam / (cam_max + 1e-8)

        return cam
