import numpy as np
import torch
import torch.nn.functional as F

from vit_model import attention_rollout


def vit_explain(model, img_tensor):
    model.eval()
    with torch.no_grad():
        logits, attn_maps = model(img_tensor, return_attn=True)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()
    confidence = probs[0, pred].item()

    rollout = attention_rollout(attn_maps)
    heatmap = rollout[0].numpy()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = np.array(
        torch.nn.functional.interpolate(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    )
    return pred, confidence, heatmap


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.conv[-3]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def explain(self, img_tensor):
        self.model.eval()
        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, pred].item()

        self.model.zero_grad()
        logits[0, pred].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(64, 64), mode="bilinear", align_corners=False)
        cam = cam[0, 0].numpy()
        cam = cam / (cam.max() + 1e-8)
        return pred, confidence, cam


def nearest_neighbors(query_vec, bank_vecs, bank_labels, k=5):
    dists = np.linalg.norm(bank_vecs - query_vec, axis=1)
    idx = np.argsort(dists)[:k]
    return idx, bank_labels[idx], dists[idx]
