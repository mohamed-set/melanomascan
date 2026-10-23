---
title: MelanoScan
emoji: 🔬
colorFrom: indigo
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
---

# 🔬 MelanoScan

**Dépister le mélanome en une photo.** MelanoScan est une application web qui estime, à partir d'une
image dermoscopique d'un grain de beauté, la probabilité qu'il s'agisse d'un mélanome, et montre les
zones de l'image qui ont guidé la décision de l'IA.

> Projet du cours *AI for Business* (Advanced AI Algorithms). Ce n'est pas un dispositif médical.

## Pourquoi

Le mélanome est le cancer de la peau le plus mortel, mais il se guérit dans plus de 90 % des cas
lorsqu'il est détecté tôt. Les délais pour voir un dermatologue dépassent souvent plusieurs mois.
Un outil de pré-tri permet de prioriser les lésions suspectes et d'aider les médecins généralistes.

## Résultats

Modèle évalué sur un jeu de test de patients jamais vus pendant l'entraînement.
_Tableau à remplir avec `metrics.json` après l'entraînement Kaggle._

| Seuil | Accuracy | Accuracy équilibrée | Sensibilité (mélanomes détectés) | Spécificité | Faux positifs | Faux négatifs |
|---|---|---|---|---|---|---|
| Équilibré (Youden) | | | | | | |
| Dépistage (≥ 90 % sensibilité) | | | | | | |

AUC test : …  ·  Ancien modèle (EfficientNetB0 100x100) : 83 % d'accuracy, 74 % de sensibilité.

## Ce qui a changé par rapport au projet de départ

| | Projet de départ | MelanoScan |
|---|---|---|
| Données | ~6 000 images ISIC | un ou plusieurs datasets ISIC Kaggle (HAM10000, ISIC 2019, ISIC 2020…), détectés automatiquement |
| Modèle | EfficientNetB0, 100x100 | EfficientNetV2-S (ImageNet-21k), 384x384 |
| Découpage | aléatoire par image | par patient (pas de fuite) |
| Déséquilibre | non traité | BCE pondérée |
| Seuil | 0,5 fixe | optimisé sur la validation (2 seuils) |
| Inférence | 1 passage | moyenne sur 4 retournements (TTA) |
| Explicabilité | aucune | carte Grad-CAM |
| Interface | aucune | application Gradio |

## Lancer l'application

```bash
git clone https://github.com/mohamed-set/melanomascan.git
cd melanomascan
pip install -r requirements.txt
# placer melanoscan_model.pth et model_config.json dans weights/
python app.py   # puis ouvrir http://127.0.0.1:7860
```

Voir le [guide utilisateur](docs/GUIDE_UTILISATEUR.md) pour le détail, y compris le déploiement sur
Hugging Face Spaces.

## Ré-entraîner le modèle

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mohamed-set/melanomascan/blob/main/training/train_melanoscan.ipynb)

Ouvrir [`training/train_melanoscan.ipynb`](training/train_melanoscan.ipynb) sur Kaggle (bouton ci-dessus), ajouter un ou plusieurs
datasets d'images de peau (voir le guide), activer GPU + Internet, puis « Run All » (~1 h 30 sur un GPU T4).
Le script source est `training/train_melanoscan.py` ; `training/to_notebook.py` régénère le notebook.

## Structure

```
app.py                     interface Gradio
melanoscan/core.py         chargement du modèle, prédiction, Grad-CAM
training/                  notebook d'entraînement Kaggle
weights/                   poids entraînés (à ajouter)
examples/                  images de démonstration (jeu de test)
docs/                      guide utilisateur, prompts IA, script de la vidéo
```

## Sources et crédits

- Données : archive ISIC (International Skin Imaging Collaboration), via les datasets Kaggle utilisés pour
  l'entraînement (à préciser selon ceux ajoutés au notebook), par exemple HAM10000
  (Tschandl P. et al., *The HAM10000 dataset*, Scientific Data, 2018) et ISIC 2019 / 2020
  (Rotemberg V. et al., Scientific Data, 2021).
- Modèle : EfficientNetV2 (Tan & Le, 2021), poids `tf_efficientnetv2_s.in21k_ft_in1k` de
  [timm / Hugging Face](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k).
- Bibliothèques : PyTorch, torchvision, timm, scikit-learn, Gradio.
- Projet de départ : [alaaeddine-ahriz/skin_cancer_detection](https://github.com/alaaeddine-ahriz/skin_cancer_detection).
- Grad-CAM : Selvaraju et al., 2017.
- Outils d'IA utilisés : voir [docs/PROMPTS.md](docs/PROMPTS.md).
