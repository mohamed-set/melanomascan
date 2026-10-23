"""Chargement du modèle, prédiction avec TTA et carte Grad-CAM."""
import json
import os

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "melanoscan_model.pth")
DEFAULT_CONFIG = os.path.join(ROOT, "weights", "model_config.json")


class MelanoScan:
    def __init__(self, weights=DEFAULT_WEIGHTS, config=DEFAULT_CONFIG, device=None):
        with open(config) as f:
            self.cfg = json.load(f)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = timm.create_model(self.cfg["model_name"], pretrained=False, num_classes=1)
        if os.path.exists(weights):
            state = torch.load(weights, map_location="cpu")
            self.model.load_state_dict({k: v.float() for k, v in state.items()})
            self.demo_mode = False
        else:
            # Sans poids, l'app reste utilisable pour tester l'interface, mais les scores n'ont aucun sens.
            self.demo_mode = True
        self.model.to(self.device).eval()
        size = self.cfg["img_size"]
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(self.cfg["mean"], self.cfg["std"]),
        ])

    def _tensor(self, image: Image.Image):
        return self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        """Probabilité de mélanome, moyennée sur 4 retournements (TTA)."""
        x = self._tensor(image)
        views = torch.cat([x, x.flip(3), x.flip(2), x.flip(2).flip(3)])
        return float(torch.sigmoid(self.model(views).squeeze(1).mean()))

    def gradcam(self, image: Image.Image, alpha=0.45) -> Image.Image:
        """Superpose la carte Grad-CAM : les zones rouges ont le plus influencé le score."""
        x = self._tensor(image)
        with torch.enable_grad():
            features = self.model.forward_features(x)
            features.retain_grad()
            self.model.forward_head(features).sum().backward()
        weights = features.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * features).sum(1)).squeeze(0).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        self.model.zero_grad(set_to_none=True)

        base = image.convert("RGB")
        heat = Image.fromarray(np.uint8(cam * 255)).resize(base.size, Image.BILINEAR)
        heat = np.asarray(heat, dtype=np.float32) / 255.0
        colored = np.stack([heat, 0.2 * np.ones_like(heat), 1.0 - heat], axis=-1) * 255
        blend = (1 - alpha * heat[..., None]) * np.asarray(base, dtype=np.float32) + alpha * heat[..., None] * colored
        return Image.fromarray(np.uint8(np.clip(blend, 0, 255)))
