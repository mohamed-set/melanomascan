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
# **Modèle** : ResNet-50 pré-entraîné sur ImageNet (bibliothèque `timm`), fine-tuné en 256x256.
# (EfficientNet est très lent sur les GPU T4 de Kaggle avec PyTorch 2.11 : ses convolutions « depthwise »
# tournent à ~15 images/s, contre plusieurs centaines pour ResNet.)
# **Durée** : environ 15 minutes sur GPU T4 x2 (arrêt automatique au bout de 15 min).
#
# **Choix pour limiter les faux positifs et les faux négatifs** :
# 1. On combine tous les datasets ajoutés pour avoir le plus de mélanomes possible, avec au plus 3 bénins par mélanome.
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
import psutil
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (auc, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# %% [markdown]
# ## Paramètres

# %%
CFG = {
    "model_name": "resnet50.a1_in1k",
    "img_size": 256,
    "epochs": 12,
    "batch_size": 64,
    "lr": 5e-4,
    "weight_decay": 1e-2,
    "warmup_epochs": 1,
    # Au-delà de ce ratio bénins/malins, on sous-échantillonne les bénins (vitesse + déséquilibre).
    "max_benign_per_malignant": 3,
    # Arrêt de sécurité : l'entraînement s'arrête avant de dépasser cette durée (minutes)
    "time_budget_min": 15, "io_threads": 64,
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
    pid = df_root["patient_id"].astype(str).str.strip()
    # Identifiants factices (-1, vide…) ou partagés par des centaines d'images : on les remplace
    # par l'identifiant de l'image, sinon tout le dataset tomberait dans un seul « patient ».
    counts = pid.map(pid.value_counts())
    invalid = pid.str.lower().isin(["", "nan", "none", "-1", "0", "unknown"]) | (counts > 300)
    df_root["patient_id"] = df_root["source"] + ":" + pid.where(~invalid, df_root["image_name"])
    return df_root


def dataset_roots(base):
    """Un dossier par dataset, que Kaggle monte /kaggle/input/<nom>, /kaggle/input/datasets/<user>/<nom>
    ou /kaggle/input/competitions/<nom>."""
    roots = []
    for entry in sorted(glob.glob(os.path.join(base, "*"))):
        name = os.path.basename(entry)
        if name in ("datasets", "notebooks", "models"):
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
    raise RuntimeError(
        "Aucun dataset d'images exploitable dans /kaggle/input. Dans « Add Input », cliquez sur le filtre "
        "« Datasets » (pas « Notebooks ») et ajoutez par exemple « Skin Cancer MNIST: HAM10000 » ou « ISIC 2019 ».")

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
def group_split(frame, held_frac, seed):
    """Met de côté ~held_frac des images, par groupes entiers, en gardant la proportion de mélanomes :
    les patients avec et sans mélanome sont tirés séparément."""
    rng = np.random.default_rng(seed)
    groups = frame.groupby("patient_id")["target"].agg(["size", "max"])
    held = []
    for _, part in groups.groupby("max"):
        order = part.index.values[rng.permutation(len(part))]
        cumulative = part.loc[order, "size"].cumsum().values
        held += list(order[cumulative <= held_frac * part["size"].sum()])
    is_held = frame["patient_id"].isin(set(held))
    return frame[~is_held].reset_index(drop=True), frame[is_held].reset_index(drop=True)


trainval_df, test_df = group_split(df, held_frac=0.15, seed=CFG["seed"])
train_df, val_df = group_split(trainval_df, held_frac=0.15 / 0.85, seed=CFG["seed"] + 1)
assert not set(train_df.patient_id) & set(test_df.patient_id)
for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{name:5s} : {len(part):6d} images, {part.target.mean():.3f} de mélanomes")

# %% [markdown]
# ## Pipeline d'images et augmentations
#
# Sur Kaggle, les fichiers de « Input » sont lus à travers le réseau : lire les images une par une
# pendant l'entraînement plafonne à ~15 images/s. On les charge donc **une seule fois**, en parallèle
# (64 lectures simultanées), réduites en 256x256 et gardées en mémoire (uint8, ~4 Go pour 20 000 images).
# Ensuite chaque époque ne prend que quelques dizaines de secondes.
#
# Le CPU ne fait que le recadrage et les retournements. Les augmentations de couleur, les rotations
# de 90° et la normalisation sont faites sur le GPU, beaucoup plus rapide.

# %%
from concurrent.futures import ThreadPoolExecutor


def read_image(path):
    size = CFG["img_size"]
    try:
        image = Image.open(path)
        image.draft("RGB", (size, size))  # décodage JPEG rapide des grandes images
        image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    except Exception:
        # Une image corrompue ne doit pas arrêter tout l'entraînement
        image = Image.new("RGB", (size, size), (128, 128, 128))
    return np.asarray(image, dtype=np.uint8)


def load_images(frame, name):
    start = time.time()
    cache = np.empty((len(frame), CFG["img_size"], CFG["img_size"], 3), dtype=np.uint8)
    with ThreadPoolExecutor(CFG["io_threads"]) as pool:
        for i, array in enumerate(pool.map(read_image, frame["path"].tolist(), chunksize=16)):
            cache[i] = array
            if (i + 1) % 2000 == 0:
                print(f"  {name} : {i + 1}/{len(frame)} images chargées ({(i + 1) / (time.time() - start):.0f} img/s)", flush=True)
    print(f"{name} : {len(frame)} images en mémoire en {time.time() - start:.0f}s", flush=True)
    return cache


train_images = load_images(train_df, "train")
val_images = load_images(val_df, "val")
test_images = load_images(test_df, "test")

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(CFG["img_size"], scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
eval_tf = None
MEAN_T = torch.tensor(MEAN, device=DEVICE).view(1, 3, 1, 1)
STD_T = torch.tensor(STD, device=DEVICE).view(1, 3, 1, 1)


def to_model_input(images, train=False):
    """uint8 (B,3,H,W) -> float normalisé sur le GPU, avec augmentations si train=True."""
    x = images.to(DEVICE, non_blocking=True).float().div_(255)
    if train:
        rand = lambda: torch.rand(x.shape[0], 1, 1, 1, device=DEVICE)
        x = x * (0.8 + 0.4 * rand())                                   # luminosité ±20 %
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * (0.8 + 0.4 * rand()) + mean                   # contraste ±20 %
        gray = x.mean(dim=1, keepdim=True)
        x = (gray + (x - gray) * (0.85 + 0.3 * rand())).clamp_(0, 1)   # saturation ±15 %
        x = torch.rot90(x, int(torch.randint(4, (1,))), dims=(2, 3))   # rotation 0/90/180/270°
    x = (x - MEAN_T) / STD_T
    return x.contiguous(memory_format=torch.channels_last if CHANNELS_LAST else torch.contiguous_format)


class LesionDataset(Dataset):
    def __init__(self, images, frame, tf):
        self.images = images
        self.targets = frame["target"].values.astype(np.float32)
        self.tf = tf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = torch.from_numpy(self.images[i]).permute(2, 0, 1)  # uint8 (3,H,W)
        if self.tf is not None:
            image = self.tf(image)
        return image.contiguous(), self.targets[i]


def make_loader(images, frame, tf, shuffle):
    return DataLoader(LesionDataset(images, frame, tf), batch_size=CFG["batch_size"], shuffle=shuffle,
                      num_workers=CFG["num_workers"], pin_memory=True, drop_last=shuffle,
                      persistent_workers=CFG["num_workers"] > 0)


train_loader = make_loader(train_images, train_df, train_tf, shuffle=True)
val_loader = make_loader(val_images, val_df, eval_tf, shuffle=False)
test_loader = make_loader(test_images, test_df, eval_tf, shuffle=False)

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
torch.backends.cudnn.benchmark = True


def benchmark(net, channels_last, amp, steps=8):
    """Images/s d'une étape d'entraînement (avant + arrière) sur des données aléatoires."""
    x = torch.randn(CFG["batch_size"], 3, CFG["img_size"], CFG["img_size"], device=DEVICE)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    for i in range(steps + 2):
        if i == 2:
            torch.cuda.synchronize()
            t = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            net(x).float().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return steps * CFG["batch_size"] / (time.time() - t)


# Choix automatique de la configuration la plus rapide sur ce GPU (quelques secondes)
CHANNELS_LAST, use_amp, use_dp = False, DEVICE.type == "cuda", False
if DEVICE.type == "cuda":
    results, initial_state = {}, {k: v.clone() for k, v in model.state_dict().items()}
    for cl in (False, True):
        model = model.to(memory_format=torch.channels_last if cl else torch.contiguous_format)
        for amp in (True, False):
            results[(cl, amp, False)] = benchmark(model, cl, amp)
    CHANNELS_LAST, use_amp, _ = max(results, key=results.get)
    model = model.to(memory_format=torch.channels_last if CHANNELS_LAST else torch.contiguous_format)
    if torch.cuda.device_count() > 1:
        dp = nn.DataParallel(model)
        results[(CHANNELS_LAST, use_amp, True)] = benchmark(dp, CHANNELS_LAST, use_amp)
        use_dp = results[(CHANNELS_LAST, use_amp, True)] > results[(CHANNELS_LAST, use_amp, False)]
    model.load_state_dict(initial_state)  # le test a modifié les statistiques BatchNorm
    for (cl, amp, dp_), speed in results.items():
        print(f"  channels_last={cl!s:5} fp16={amp!s:5} 2 GPU={dp_!s:5} : {speed:5.0f} img/s")
    print(f"Choisi : channels_last={CHANNELS_LAST}, fp16={use_amp}, 2 GPU={use_dp}", flush=True)
if use_dp:
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
        images = to_model_input(images)
        views = [images]
        if tta:
            views += [images.flip(3), images.flip(2), images.flip(2).flip(3)]
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            logits = torch.stack([model(v).squeeze(1).float() for v in views]).mean(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


best_auc, history = 0.0, []
train_start = time.time()
best_path = os.path.join(CFG["output_dir"], "best_model.pth")
for epoch in range(CFG["epochs"]):
    model.train()
    start, running = time.time(), 0.0
    for step, (images, y) in enumerate(train_loader):
        images, y = to_model_input(images, train=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            loss = criterion(model(images).squeeze(1), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running += loss.item()
        if (step + 1) in (20, 100) or (step + 1) % 200 == 0:
            ram = psutil.virtual_memory()
            print(f"  étape {step + 1}/{len(train_loader)} | {(step + 1) * CFG['batch_size'] / (time.time() - start):.0f} img/s"
                  f" | RAM {ram.used / 1e9:.1f}/{ram.total / 1e9:.1f} Go", flush=True)
    val_probs, val_targets = predict(val_loader)
    val_auc = roc_auc_score(val_targets, val_probs)
    history.append({"epoch": epoch + 1, "train_loss": running / len(train_loader), "val_auc": val_auc})
    print(f"Époque {epoch + 1}/{CFG['epochs']} | loss {running / len(train_loader):.4f} | "
          f"val AUC {val_auc:.4f} | {time.time() - start:.0f}s", flush=True)
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save((model.module if hasattr(model, "module") else model).state_dict(), best_path)
    elapsed = time.time() - train_start
    if elapsed + elapsed / (epoch + 1) > CFG["time_budget_min"] * 60:
        print(f"Budget de {CFG['time_budget_min']} min atteint : arrêt après {epoch + 1} époques.", flush=True)
        break

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
# - `melanoscan_model.pth` : poids du modèle en float16 (~47 Mo)
# - `model_config.json` : nom du modèle, taille d'image et seuils
# - `metrics.json`, `test_results.png` : résultats pour le README et la vidéo
# - `examples/` : quelques images du jeu de test pour la démo

# %%
out = CFG["output_dir"]
state = (model.module if hasattr(model, "module") else model).state_dict()
torch.save({k: v.half().cpu() for k, v in state.items()}, os.path.join(out, "melanoscan_model.pth"))
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
