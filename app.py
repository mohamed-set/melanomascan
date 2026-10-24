"""MelanoScan : interface Gradio pour estimer le risque de mélanome sur une photo de lésion."""
import glob
import os

import gradio as gr

from melanoscan.core import ROOT, MelanoScan

scanner = MelanoScan()
LOW = min(scanner.cfg["threshold_screening"], scanner.cfg["threshold_balanced"])
HIGH = scanner.cfg.get("threshold_high", max(scanner.cfg["threshold_screening"], scanner.cfg["threshold_balanced"]))

DISCLAIMER = (
    "⚠️ **MelanoScan est un projet étudiant, pas un dispositif médical.** "
    "Le résultat ne remplace jamais l'avis d'un dermatologue. "
    "Au moindre doute (lésion qui change, saigne, ou est asymétrique), consultez."
)


def analyse(image, show_cam):
    if image is None:
        raise gr.Error("Ajoutez d'abord une photo de la lésion.")
    prob = scanner.predict(image)
    if prob >= HIGH:
        level, advice = "🔴 Risque élevé", "Prenez rendez-vous rapidement avec un dermatologue."
    elif prob >= LOW:
        level, advice = "🟠 À surveiller", "Faites contrôler cette lésion lors d'une consultation."
    else:
        level, advice = "🟢 Risque faible", "Continuez à surveiller la lésion (règle ABCDE)."
    summary = f"## {level}\n**Score de mélanome : {prob:.0%}**\n\n{advice}"
    if scanner.demo_mode:
        summary += "\n\n_Mode démo : aucun poids entraîné trouvé dans `weights/`, le score est aléatoire._"
    cam = scanner.gradcam(image) if show_cam else None
    return {"Mélanome": prob, "Bénin": 1 - prob}, summary, cam


examples = sorted(glob.glob(os.path.join(ROOT, "examples", "*.jpg")))

with gr.Blocks(title="MelanoScan") as demo:
    gr.Markdown(
        "# 🔬 MelanoScan\n"
        "Déposez une photo dermoscopique d'un grain de beauté : l'IA estime le risque de mélanome "
        "et montre les zones de l'image qui ont guidé sa décision."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Photo de la lésion", sources=["upload", "webcam", "clipboard"])
            show_cam = gr.Checkbox(value=True, label="Afficher la carte d'attention (Grad-CAM)")
            button = gr.Button("Analyser", variant="primary")
            if examples:
                gr.Examples(examples, inputs=image, label="Exemples (jeu de test ISIC)")
        with gr.Column():
            summary = gr.Markdown()
            scores = gr.Label(label="Probabilités", num_top_classes=2)
            cam = gr.Image(label="Zones qui ont influencé le modèle", interactive=False)
    gr.Markdown(DISCLAIMER)
    with gr.Accordion("Comment fonctionne MelanoScan ?", open=False):
        gr.Markdown(
            f"- Modèle : `{scanner.cfg['model_name']}` fine-tuné sur ~50 000 images ISIC 2019 + 2020.\n"
            f"- AUC sur le jeu de test : **{scanner.cfg.get('test_auc', 'n/a')}**.\n"
            f"- Score ≥ {HIGH:.0%} : risque élevé (seuil équilibré entre faux positifs et faux négatifs).\n"
            f"- Score entre {LOW:.0%} et {HIGH:.0%} : à surveiller (seuil de dépistage, 90 % des mélanomes au-dessus).\n"
            "- Chaque image est analysée 4 fois (retournements) et les scores sont moyennés."
        )
    button.click(analyse, inputs=[image, show_cam], outputs=[scores, summary, cam])

if __name__ == "__main__":
    demo.launch()
