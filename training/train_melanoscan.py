# %% [markdown]
# # MelanoScan : entraînement du classifieur de mélanome
#
# Ce notebook entraîne le modèle utilisé par l'application MelanoScan.
# Il est fait pour tourner sur **Kaggle** (GPU T4 ou P100, Internet activé).
#
# **Données** (à ajouter au notebook via « Add Input ») :
# - `cdeotte/jpeg-melanoma-384x384` : ISIC 2020, images redimensionnées en 384x384
# - `cdeotte/jpeg-isic2019-384x384` : ISIC 2019, images redimensionnées en 384x384
#
# **Modèle** : EfficientNetV2-S pré-entraîné sur ImageNet-21k (bibliothèque `timm`),
# fine-tuné entièrement en 384x384.
#
# **Choix pour limiter les faux positifs et les faux négatifs** :
# 1. On combine ISIC 2019 et 2020 pour avoir environ 5 000 mélanomes au lieu de 584.
# 2. Le découpage train/val/test se fait **par patient** pour éviter les fuites.
# 3. La perte BCE est pondérée pour compenser le déséquilibre des classes.
# 4. Le seuil de décision est choisi sur la validation (indice de Youden), pas fixé à 0,5.
# 5. On moyenne les prédictions sur 4 retournements de l'image (TTA).

# %%
!pip install -q "timm>=1.0"

# %%
import glob
import json
import math
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (auc, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# %% [markdown]
# ## Paramètres

# %%
CFG = {
    "model_name": "tf_efficientnetv2_s.in21k_ft_in1k",
    "img_size": 384,
    "epochs": 10,
    "batch_size": 32,
    "lr": 2e-4,
    "weight_decay": 1e-2,
    "warmup_epochs": 1,
    # ISIC 2020 contient ~32 000 images bénignes : on en garde une partie pour
    # accélérer l'entraînement et réduire le déséquilibre.
    "benign_keep_frac_2020": 0.5,
    "num_workers": os.cpu_count() or 2,
    "seed": 42,
    "target_sensitivity": 0.90,
    "input_dir": "/kaggle/input",
    "output_dir": "/kaggle/working",
    # Mettre à True pour un essai rapide (quelques centaines d'images, 1 époque).
    "debug": False,
}
if CFG["debug"]:
    CFG.update(epochs=1, img_size=128)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", DEVICE, "| GPUs :", torch.cuda.device_count())

# %% [markdown]
# ## Chargement des métadonnées
#
# On cherche les fichiers `train.csv` des deux datasets dans `/kaggle/input`,
# quel que soit le chemin exact de montage.

# %%
def find_dataset(name):
    hits = glob.glob(os.path.join(CFG["input_dir"], "**", name, "train.csv"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"Dataset '{name}' introuvable : ajoutez-le via « Add Input ».")
    return os.path.dirname(hits[0])


def load_source(name, source):
    root = find_dataset(name)
    df = pd.read_csv(os.path.join(root, "train.csv"))
    # tfrecord == -1 marque les doublons dans les datasets de Chris Deotte
    if "tfrecord" in df.columns:
        df = df[df["tfrecord"] != -1]
    df["path"] = df["image_name"].map(lambda n: os.path.join(root, "train", f"{n}.jpg"))
    df["source"] = source
    df["patient_id"] = df["patient_id"].fillna(df["image_name"])
    return df[["image_name", "patient_id", "target", "path", "source"]]


df2020 = load_source("jpeg-melanoma-384x384", "isic2020")
df2019 = load_source("jpeg-isic2019-384x384", "isic2019")

benign2020 = df2020[df2020.target == 0].sample(frac=CFG["benign_keep_frac_2020"], random_state=CFG["seed"])
df2020 = pd.concat([benign2020, df2020[df2020.target == 1]])
df = pd.concat([df2020, df2019], ignore_index=True)
df = df[df["path"].map(os.path.exists)].reset_index(drop=True)

if CFG["debug"]:
    df = pd.concat([df[df.target == t].head(200) for t in (0, 1)]).reset_index(drop=True)

print(df.groupby(["source", "target"]).size().unstack())
print("Total :", len(df), "| part de mélanomes :", round(df.target.mean(), 3))

# %% [markdown]
# ## Découpage train / validation / test par patient
#
# Toutes les images d'un même patient restent dans le même ensemble.

# %%
def group_split(frame, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(sgkf.split(frame, frame.target, frame.patient_id))
    return frame.iloc[rest_idx].reset_index(drop=True), frame.iloc[held_idx].reset_index(drop=True)


trainval_df, test_df = group_split(df, n_splits=7, seed=CFG["seed"])     # ~14 % test
train_df, val_df = group_split(trainval_df, n_splits=6, seed=CFG["seed"])  # ~14 % val
assert not set(train_df.patient_id) & set(test_df.patient_id)
for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{name:5s} : {len(part):6d} images, {part.target.mean():.3f} de mélanomes")

# %% [markdown]
# ## Pipeline d'images et augmentations

# %%
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(CFG["img_size"], scale=(0.6, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class LesionDataset(Dataset):
    def __init__(self, frame, tf):
        self.paths = frame["path"].values
        self.targets = frame["target"].values.astype(np.float32)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.tf(image), self.targets[i]


def make_loader(frame, tf, shuffle):
    return DataLoader(LesionDataset(frame, tf), batch_size=CFG["batch_size"], shuffle=shuffle,
                      num_workers=CFG["num_workers"], pin_memory=True, drop_last=shuffle)


train_loader = make_loader(train_df, train_tf, shuffle=True)
val_loader = make_loader(val_df, eval_tf, shuffle=False)
test_loader = make_loader(test_df, eval_tf, shuffle=False)

# %% [markdown]
# ## Modèle, perte et optimiseur
#
# La pondération `pos_weight = sqrt(négatifs / positifs)` donne plus de poids aux
# mélanomes sans sur-corriger (une correction complète ferait exploser les faux positifs).

# %%
model = timm.create_model(CFG["model_name"], pretrained=True, num_classes=1, drop_rate=0.3, drop_path_rate=0.2)
model = model.to(DEVICE)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

n_pos = train_df.target.sum()
pos_weight = math.sqrt((len(train_df) - n_pos) / n_pos)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

total_steps = CFG["epochs"] * len(train_loader)
warmup_steps = CFG["warmup_epochs"] * len(train_loader)


def lr_lambda(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
print(f"pos_weight = {pos_weight:.2f}")

# %% [markdown]
# ## Boucle d'entraînement
#
# On garde le modèle qui a la meilleure AUC de validation.

# %%
@torch.no_grad()
def predict(loader, tta=False):
    model.eval()
    probs, targets = [], []
    for images, y in loader:
        images = images.to(DEVICE, non_blocking=True)
        views = [images]
        if tta:
            views += [images.flip(3), images.flip(2), images.flip(2).flip(3)]
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            logits = torch.stack([model(v).squeeze(1).float() for v in views]).mean(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


best_auc, history = 0.0, []
best_path = os.path.join(CFG["output_dir"], "best_model.pth")
for epoch in range(CFG["epochs"]):
    model.train()
    start, running = time.time(), 0.0
    for step, (images, y) in enumerate(train_loader):
        images, y = images.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            loss = criterion(model(images).squeeze(1), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running += loss.item()
    val_probs, val_targets = predict(val_loader)
    val_auc = roc_auc_score(val_targets, val_probs)
    history.append({"epoch": epoch + 1, "train_loss": running / len(train_loader), "val_auc": val_auc})
    print(f"Époque {epoch + 1}/{CFG['epochs']} | loss {running / len(train_loader):.4f} | "
          f"val AUC {val_auc:.4f} | {time.time() - start:.0f}s")
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save((model.module if hasattr(model, "module") else model).state_dict(), best_path)

(model.module if hasattr(model, "module") else model).load_state_dict(torch.load(best_path))
print("Meilleure AUC de validation :", round(best_auc, 4))

# %% [markdown]
# ## Choix du seuil de décision sur la validation
#
# - **Seuil équilibré (Youden)** : maximise sensibilité + spécificité, c'est-à-dire
#   le meilleur compromis entre faux positifs et faux négatifs. C'est le seuil par défaut de l'app.
# - **Seuil « dépistage »** : garantit au moins 90 % de mélanomes détectés, au prix de plus de faux positifs.

# %%
val_probs, val_targets = predict(val_loader, tta=True)
fpr, tpr, thresholds = roc_curve(val_targets, val_probs)
youden_threshold = float(thresholds[np.argmax(tpr - fpr)])
screening_threshold = float(thresholds[np.argmax(tpr >= CFG["target_sensitivity"])])
print(f"Seuil équilibré : {youden_threshold:.3f} | seuil dépistage : {screening_threshold:.3f}")

# %% [markdown]
# ## Évaluation finale sur le jeu de test (jamais vu pendant l'entraînement)

# %%
test_probs, test_targets = predict(test_loader, tta=True)


def report(threshold):
    pred = (test_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_targets, pred).ravel()
    return {
        "threshold": round(threshold, 4),
        "accuracy": round((tp + tn) / len(pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(test_targets, pred), 4),
        "sensitivity": round(tp / (tp + fn), 4),
        "specificity": round(tn / (tn + fp), 4),
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn),
    }


metrics = {
    "test_auc": round(roc_auc_score(test_targets, test_probs), 4),
    "n_test": int(len(test_targets)),
    "n_test_melanoma": int(test_targets.sum()),
    "balanced": report(youden_threshold),
    "screening": report(screening_threshold),
    "default_0.5": report(0.5),
    "history": history,
}
print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))

# %% [markdown]
# ## Graphiques pour la démo

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
t_fpr, t_tpr, _ = roc_curve(test_targets, test_probs)
axes[0].plot(t_fpr, t_tpr, label=f"AUC = {auc(t_fpr, t_tpr):.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="grey")
axes[0].set(title="Courbe ROC (test)", xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs")
axes[0].legend(loc="lower right")
cm = confusion_matrix(test_targets, (test_probs >= youden_threshold).astype(int))
axes[1].imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    axes[1].text(j, i, str(v), ha="center", va="center", fontsize=14,
                 color="white" if v > cm.max() / 2 else "black")
axes[1].set(title="Matrice de confusion (seuil équilibré)", xticks=[0, 1], yticks=[0, 1],
            xticklabels=["Bénin", "Mélanome"], yticklabels=["Bénin", "Mélanome"],
            xlabel="Prédit", ylabel="Réel")
plt.tight_layout()
plt.savefig(os.path.join(CFG["output_dir"], "test_results.png"), dpi=150)
plt.show()

# %% [markdown]
# ## Export pour l'application
#
# Fichiers produits dans `/kaggle/working` (onglet « Output ») :
# - `melanoscan_model.pth` : poids du modèle en float16 (~40 Mo)
# - `model_config.json` : nom du modèle, taille d'image et seuils
# - `metrics.json`, `test_results.png` : résultats pour le README et la vidéo
# - `examples/` : quelques images du jeu de test pour la démo

# %%
out = CFG["output_dir"]
state = (model.module if hasattr(model, "module") else model).state_dict()
torch.save({k: v.half() for k, v in state.items()}, os.path.join(out, "melanoscan_model.pth"))
os.remove(best_path)

config = {
    "model_name": CFG["model_name"],
    "img_size": CFG["img_size"],
    "mean": MEAN,
    "std": STD,
    "threshold_balanced": youden_threshold,
    "threshold_screening": screening_threshold,
    "test_auc": metrics["test_auc"],
}
with open(os.path.join(out, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)
with open(os.path.join(out, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Exemples pour la démo : les plus nets de chaque classe, plus un cas difficile
examples_dir = os.path.join(out, "examples")
os.makedirs(examples_dir, exist_ok=True)
scored = test_df.assign(prob=test_probs)
picks = pd.concat([
    scored[scored.target == 1].nlargest(3, "prob").assign(kind="melanome"),
    scored[scored.target == 0].nsmallest(3, "prob").assign(kind="benin"),
    scored[(scored.target == 1) & (scored.prob < youden_threshold)].head(1).assign(kind="melanome_difficile"),
])
for _, row in picks.iterrows():
    shutil.copy(row.path, os.path.join(examples_dir, f"{row.kind}_{row.image_name}.jpg"))
print(sorted(os.listdir(out)))
