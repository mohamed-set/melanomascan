"""Exporte le modèle entraîné en ONNX pour la web app (inférence dans le navigateur).

Usage : python training/export_onnx.py [weights/melanoscan_model.pth] [weights/model_config.json] [docs/model]

Le modèle exporté renvoie deux sorties :
- prob : probabilité de mélanome, forme (B,)
- cam  : carte d'activation de classe (CAM), forme (B, h, w). Comme la tête du réseau est
  « moyenne globale puis couche linéaire », CAM = somme des cartes de features pondérées par
  les poids de la couche linéaire : c'est exact, sans gradient.
"""
import json
import os
import shutil
import sys

import timm
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "weights", "melanoscan_model.pth")
config_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "weights", "model_config.json")
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "docs", "model")

with open(config_path) as f:
    cfg = json.load(f)
model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=1)
if os.path.exists(weights):
    model.load_state_dict({k: v.float() for k, v in torch.load(weights, map_location="cpu").items()})
else:
    print(f"ATTENTION : {weights} introuvable, export d'un modèle non entraîné (test uniquement).")
model.eval()


class WithCam(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.fc = net.get_classifier()

    def forward(self, x):
        feats = self.net.forward_features(x)
        logit = self.net.forward_head(feats).squeeze(1)
        cam = torch.einsum("bchw,c->bhw", feats, self.fc.weight[0]) + self.fc.bias[0]
        return torch.sigmoid(logit), cam


os.makedirs(out_dir, exist_ok=True)
size = cfg["img_size"]
onnx_path = os.path.join(out_dir, "melanoscan.onnx")
torch.onnx.export(
    WithCam(model), torch.randn(1, 3, size, size), onnx_path,
    input_names=["image"], output_names=["prob", "cam"],
    dynamic_axes={"image": {0: "batch"}, "prob": {0: "batch"}, "cam": {0: "batch"}},
    opset_version=17, dynamo=False,
)
# Poids en float16 : fichier deux fois plus léger pour le navigateur, entrées/sorties restent en float32
try:
    import onnx
    from onnxconverter_common import float16
    onnx.save(float16.convert_float_to_float16(onnx.load(onnx_path), keep_io_types=True), onnx_path)
except ImportError:
    print("onnxconverter-common absent : modèle laissé en float32 (pip install onnxconverter-common)")
shutil.copy(config_path, os.path.join(out_dir, "model_config.json"))
print(f"{onnx_path} : {os.path.getsize(onnx_path) / 1e6:.1f} Mo")
