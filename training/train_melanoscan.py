# %% [markdown]
# # MelanoScan : entraînement du classifieur de mélanome
#
# Ce notebook entraîne le modèle utilisé par l'application MelanoScan.
# Il est fait pour tourner sur **Kaggle** (GPU T4 ou P100, Internet activé).
#
# **Données** : ajoutez **un ou plusieurs** datasets d'images de peau via « Add Input ».
# Le notebook détecte tout seul leur format. Formats reconnus :
# - un CSV avec une colonne d'identifiant d'image (`image_name`, `image_id`, `isic_id`…) et une
#   colonne de label (`target`, `benign_malignant`, `dx`, `MEL`, `diagnosis`, `label`) ;
#   exemples : HAM10000 (« Skin Cancer MNIST: HAM10000 »), ISIC 2019, ISIC 2020 / SIIM-ISIC ;
# - des dossiers d'images nommés `benign` / `malignant` (ou `melanoma`, `nevus`…) ;
#   exemples : « Melanoma Skin Cancer Dataset of 10000 Images », « Skin Cancer: Malignant vs. Benign ».
#
# Plus il y a d'images de mélanomes, meilleur sera le modèle : combinez plusieurs datasets si possible.
# Les doublons (même identifiant ISIC) sont supprimés automatiquement.
#
# **Modèle** : EfficientNetV2-S pré-entraîné sur ImageNet-21k (bibliothèque `timm`),
# fine-tuné entièrement en 384x384.
#
# **Choix pour limiter les faux positifs et les faux négatifs** :
# 1. On combine tous les datasets ajoutés pour avoir le plus de mélanomes possible.
# 2. Le découpage train/val/test se fait **par patient ou par lésion** quand l'info existe, pour éviter les fuites.
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
    # Au-delà de ce ratio bénins/malins, on sous-échantillonne les bénins (vitesse + déséquilibre).
    "max_benign_per_malignant": 8,
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
if DEVICE.type == "cuda":
    major, minor = torch.cuda.get_device_capability(0)
    print("GPU :", torch.cuda.get_device_name(0))
    if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
        raise RuntimeError("Ce GPU n'est pas pris en charge par la version de PyTorch de Kaggle. "
                           "Settings → Accelerator → choisissez « GPU T4 x2 » puis relancez.")
else:
    print("ATTENTION : pas de GPU. Settings → Accelerator → « GPU T4 x2 », sinon l'entraînement prendra des jours.")

# %% [markdown]
# ## Détection et chargement des datasets
#
# Pour chaque dataset présent dans `/kaggle/input`, on cherche d'abord un CSV avec des labels.
# À défaut, on lit le label dans le nom du dossier de chaque image.

# %%
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")
ID_COLS = ["image_name", "image_id", "isic_id", "image", "filename", "file", "id"]
GROUP_COLS = ["patient_id", "lesion_id"]
MALIGNANT_DIRS = {"malignant", "melanoma", "mel", "cancer"}
BENIGN_DIRS = {"benign", "nevus", "nv", "non_melanoma", "not_melanoma", "normal"}


def csv_labels(frame):
    """Renvoie une série 0/1 (1 = mélanome/malin) ou None si le CSV n'a pas de label exploitable."""
    cols = {c.lower(): c for c in frame.columns}
    if "target" in cols:
        return pd.to_numeric(frame[cols["target"]], errors="coerce")
    if "mel" in cols and pd.api.types.is_numeric_dtype(frame[cols["mel"]]):
        return (frame[cols["mel"]] > 0.5).astype(float)
    if "dx" in cols:
        return frame[cols["dx"]].astype(str).str.lower().eq("mel").astype(float)
    if "benign_malignant" in cols:
        return frame[cols["benign_malignant"]].astype(str).str.lower().map({"malignant": 1.0, "benign": 0.0})
    if "diagnosis" in cols:
        return frame[cols["diagnosis"]].astype(str).str.lower().str.contains("melanoma").astype(float)
    if "label" in cols:
        lab = frame[cols["label"]]
        if pd.api.types.is_numeric_dtype(lab):
            return lab.astype(float)
        return lab.astype(str).str.lower().isin(MALIGNANT_DIRS).astype(float)
    return None


def load_dataset(root):
    images = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMAGE_EXT):
                images.setdefault(os.path.splitext(f)[0], os.path.join(dirpath, f))
    if not images:
        return None

    parts = []
    for csv_path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        # sample_submission.csv et test.csv ont des labels factices ou absents
        if any(k in os.path.basename(csv_path).lower() for k in ("submission", "test")):
            continue
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        cols = {c.lower(): c for c in frame.columns}
        id_col = next((cols[c] for c in ID_COLS if c in cols), None)
        labels = csv_labels(frame)
        if id_col is None or labels is None:
            continue
        if "tfrecord" in cols:  # doublons marqués -1 dans les datasets de Chris Deotte
            keep = frame[cols["tfrecord"]] != -1
            frame, labels = frame[keep], labels[keep]
        stems = frame[id_col].astype(str).map(lambda s: os.path.splitext(os.path.basename(s))[0])
        group_col = next((cols[c] for c in GROUP_COLS if c in cols), None)
        part = pd.DataFrame({
            "image_name": stems.values,
            "target": labels.values,
            "patient_id": frame[group_col].values if group_col else stems.values,
        })
        part["path"] = part["image_name"].map(images)
        part = part.dropna(subset=["path", "target"])
        if len(part):
            parts.append(part)

    if not parts:  # pas de CSV exploitable : label = nom du dossier parent
        rows = []
        for stem, path in images.items():
            folders = {d.lower() for d in path.split(os.sep)[-3:-1]}
            if folders & MALIGNANT_DIRS:
                rows.append((stem, 1.0, stem, path))
            elif folders & BENIGN_DIRS:
                rows.append((stem, 0.0, stem, path))
        if rows:
            parts.append(pd.DataFrame(rows, columns=["image_name", "target", "patient_id", "path"]))

    if not parts:
        return None
    df_root = pd.concat(parts, ignore_index=True)
    df_root["source"] = os.path.basename(root.rstrip(os.sep))
    df_root["patient_id"] = df_root["patient_id"].fillna(df_root["image_name"]).astype(str)
    return df_root


def dataset_roots(base):
    """Un dossier par dataset, que Kaggle monte /kaggle/input/<nom>, /kaggle/input/datasets/<user>/<nom>
    ou /kaggle/input/competitions/<nom>."""
    roots = []
    for entry in sorted(glob.glob(os.path.join(base, "*"))):
        name = os.path.basename(entry)
        if name == "datasets":
            roots += sorted(glob.glob(os.path.join(entry, "*", "*")))
        elif name == "competitions":
            roots += sorted(glob.glob(os.path.join(entry, "*")))
        else:
            roots.append(entry)
    return [r for r in roots if os.path.isdir(r)]


frames = []
for root in dataset_roots(CFG["input_dir"]):
    found = load_dataset(root)
    if found is None:
        print(f"Ignoré (aucune image labellisée) : {root}")
    else:
        print(f"OK : {root} -> {len(found)} images, {int(found.target.sum())} mélanomes/malins")
        frames.append(found)
if not frames:
    raise RuntimeError("Aucun dataset exploitable : ajoutez un dataset d'images de peau via « Add Input ».")

df = pd.concat(frames, ignore_index=True)
df = df[df["target"].isin([0, 1])]
df["target"] = df["target"].astype(int)
df = df.drop_duplicates(subset="image_name").reset_index(drop=True)

# Limite le nombre d'images bénignes pour accélérer l'entraînement si le déséquilibre est extrême
n_pos = int(df.target.sum())
max_benign = CFG["max_benign_per_malignant"] * n_pos
if (df.target == 0).sum() > max_benign:
    df = pd.concat([df[df.target == 1],
                    df[df.target == 0].sample(max_benign, random_state=CFG["seed"])]).reset_index(drop=True)

if CFG["debug"]:
    df = pd.concat([df[df.target == t].head(200) for t in (0, 1)]).reset_index(drop=True)

print(df.groupby(["source", "target"]).size().unstack(fill_value=0))
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
        try:
            image = Image.open(self.paths[i])
            image.draft("RGB", (CFG["img_size"] * 2, CFG["img_size"] * 2))  # décodage JPEG rapide des grandes images
            image = image.convert("RGB")
        except Exception:
            # Une image corrompue ne doit pas arrêter tout l'entraînement
            image = Image.new("RGB", (CFG["img_size"], CFG["img_size"]))
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
try:
    model = timm.create_model(CFG["model_name"], pretrained=True, num_classes=1, drop_rate=0.3, drop_path_rate=0.2)
except Exception as e:
    raise RuntimeError("Téléchargement des poids pré-entraînés impossible : activez Settings → Internet "
                       "(compte Kaggle vérifié par téléphone requis).") from e
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
    tn, fp, fn, tp = confusion_matrix(test_targets, pred, labels=[0, 1]).ravel()
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
