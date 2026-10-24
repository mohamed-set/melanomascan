# Utilisation d'outils d'IA générative

Le sujet demande de montrer où et comment nous avons utilisé des outils d'IA, avec les prompts.

| Étape | Outil | Prompt (résumé ou texte exact) | Ce que nous avons gardé |
|---|---|---|---|
| Choix du projet | Claude (Anthropic) | « J'ai un projet dans la matière AI for business. Vérifie sur mon GitHub, je pense que le projet de classification binaire de mélanome pourrait être une bonne idée. Ça utilise Kaggle pour la base de données ISIC et GitHub pour le code. Le but serait de rajouter une interface utilisable. » | Analyse du sujet et du repo de départ, liste des points faibles (poids absents, données, pas d'interface) |
| Choix du modèle | Claude | « On ré-entraîne, Gradio, fais un nouveau repo à mon nom, on est 4 dans le groupe. Choisis un modèle approprié et optimisé pour maximiser l'accuracy et minimiser les faux positifs/négatifs. » | EfficientNetV2-S 384x384 (puis, après des échecs trop longs sur Kaggle, ResNet-50 256x256 en 15 min), ISIC 2019+2020, découpage par patient, BCE pondérée, seuils optimisés, TTA, Grad-CAM, notebook Kaggle, app Gradio, docs |
| … | … | … | … |

_À compléter par chaque membre du groupe avec les prompts utilisés (ChatGPT, Copilot, etc.)._
