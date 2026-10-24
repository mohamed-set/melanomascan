# Guide utilisateur MelanoScan

## 1. Utiliser l'application

1. Ouvrez l'application (lien Hugging Face Spaces, ou `http://127.0.0.1:7860` en local).
2. Déposez une photo de la lésion, prenez-la avec la webcam, ou cliquez sur un exemple.
   Les meilleurs résultats s'obtiennent avec une **image dermoscopique** centrée sur la lésion,
   comme celles des exemples. Une photo de smartphone classique donne un résultat moins fiable.
3. Cliquez sur **Analyser**.
4. Lisez le résultat :

| Niveau | Signification | Conseil affiché |
|---|---|---|
| 🟢 Risque faible | score sous le seuil de dépistage | continuer à surveiller (règle ABCDE) |
| 🟠 À surveiller | score entre les deux seuils | faire contrôler lors d'une consultation |
| 🔴 Risque élevé | score au-dessus du seuil équilibré | consulter rapidement un dermatologue |

5. La **carte Grad-CAM** colore en rouge les zones qui ont le plus pesé dans la décision. Si le rouge
   tombe hors de la lésion (poils, règle, bord noir), le résultat est moins fiable.

MelanoScan ne pose pas de diagnostic et ne remplace pas un médecin.

## 2. Installer et lancer en local

Prérequis : Python 3.10 ou plus récent.

```bash
git clone https://github.com/mohamed-set/melanomascan.git
cd melanomascan
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Placez les deux fichiers produits par l'entraînement dans `weights/` :
`melanoscan_model.pth` et `model_config.json`. Copiez aussi les images de `examples/` si vous les avez.

```bash
python app.py
```

Sans poids, l'application démarre en **mode démo** : l'interface fonctionne mais le score est aléatoire.

## 3. Déployer sur Hugging Face Spaces (lien public pour la démo)

1. Créez un Space sur https://huggingface.co/new-space (SDK : **Gradio**, matériel : CPU basic gratuit).
2. Poussez le contenu du repo dans le Space (`git push` vers l'URL du Space), poids inclus.
   Le fichier de poids fait ~50 Mo ; activez Git LFS si Hugging Face le demande (`git lfs track "*.pth"`).
3. Le Space se construit seul à partir de `requirements.txt` et du bloc en tête du `README.md`.

## 4. Ré-entraîner le modèle sur Kaggle

1. Sur Kaggle : **Create → New Notebook → File → Import Notebook**, choisissez `training/train_melanoscan.ipynb`.
2. **Add Input** (filtre **Datasets**) : ajoutez les deux datasets utilisés pour le modèle fourni :
   - « JPEG Melanoma 256x256 » de Chris Deotte (ISIC 2020, 33 126 images, 584 mélanomes, 830 Mo) ;
   - « JPEG ISIC 2019 256x256 » de Chris Deotte (25 331 images, ~4 500 mélanomes, 450 Mo).
   D'autres datasets ISIC marchent aussi (HAM10000, « Melanoma Skin Cancer Dataset of 10000 Images »…),
   mais préférez les versions 256x256 : les grandes images remplissent la mémoire de Kaggle et ralentissent tout.
   Le notebook détecte le format tout seul et affiche, pour chaque dataset, combien d'images il a reconnues.
   Un dataset marqué « Ignoré » n'a pas de label exploitable.
3. **Settings** : Accelerator = GPU T4 x2 (le P100 n'est plus pris en charge par PyTorch), Internet = On (compte vérifié par téléphone requis).
4. Pour un essai rapide, mettez `"debug": True` dans la cellule des paramètres, puis remettez `False`.
5. **Save Version → Save & Run All**. Comptez environ 20 min (entraînement limité à 15 min).
6. Dans l'onglet **Output**, téléchargez `melanoscan_model.pth`, `model_config.json`, `metrics.json`,
   `test_results.png` et le dossier `examples/`.

Le découpage est fixé par `seed = 42`, ce qui rend les résultats reproductibles à de petites variations GPU près.
