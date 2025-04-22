import os
import gradio as gr
import tensorflow as tf
import numpy as np
import random
import pickle
from tensorflow.keras.layers import TextVectorization



# load the model
model = tf.keras.models.load_model("model/last_model.keras", compile=False)

# load vectorizer
with open('model\vectorizer_vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

MAX_WORDS = 40000
SEQUENCE_LENGTH = 600

vectorizer = TextVectorization(
    max_tokens=MAX_WORDS,
    output_sequence_length=SEQUENCE_LENGTH,
    output_mode='int'
)

vectorizer.set_vocabulary(vocabulary)

# labels
labels = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']


awareness_messages = {
    'toxicity': [
        "Toxic comments can deeply hurt others—choose empathy.",
        "What you say online matters. Avoid spreading negativity.",
        "Toxicity can damage communities. Lead with kindness.",
    ],
    'severe_toxicity': [
        "This level of toxicity can be extremely harmful. Please reconsider your words.",
        "Such extreme language might get flagged or reported—stay respectful.",
        "Severe toxicity can lead to serious consequences, even legal action.",
    ],
    'obscene': [
        "Obscene language can make others uncomfortable. Let's keep it respectful.",
        "Using vulgar terms doesn't strengthen your point.",
        "Consider expressing yourself in a cleaner way.",
    ],
    'threat': [
        "Threats aren't just harmful—they can be illegal.",
        "Making others feel unsafe is never acceptable.",
        "This might be interpreted as a threat. Please stay kind.",
    ],
    'insult': [
        "Insulting others reflects poorly on you, not them.",
        "Words can wound. Please avoid using personal attacks.",
        "Disagreements are okay—insults are not.",
    ],
    'identity_attack': [
        "Targeting someone's identity promotes hate. Please don't.",
        "Disrespecting who someone is can leave lifelong scars.",
        "We are all different. Let's celebrate that—not attack it.",
    ],
    'sexual_explicit': [
        "Explicit content might not be appropriate for all audiences.",
        "Sexual language can cross boundaries—respect others' comfort zones.",
        "Please consider whether this message is suitable for public discussion.",
    ],
}

# Images links from imgur
label_to_link_image2 = {
    'toxicity': "https://i.imgur.com/LYWCuH3.png",
    'severe_toxicity': "https://i.imgur.com/YT8roF8.png",
    'obscene': "https://i.imgur.com/Rkb3sYh.png",
    'threat': "https://i.imgur.com/YmEEHJ1.png",
    'insult': "https://i.imgur.com/A4iq2Km.png",
    'identity_attack': "https://i.imgur.com/SjIAa6F.png",
    'sexual_explicit': "https://i.imgur.com/pATebIx.png"
}

# main function
def classify_and_advise(text):
    X = vectorizer([text])
    prediction = model.predict(X)[0]

    detected_cards = []
    for i, prob in enumerate(prediction):
        if prob >= 0.5:
            label = labels[i]
            img_url = label_to_link_image2.get(label)
            msg = random.choice(awareness_messages.get(label, ["Be respectful online."]))
            detected_cards.append(f"""
            <div class='card'>
                <img src="{img_url}" alt="{label}">
                <h3 class='category-title'>{label.replace('_', ' ').title()}</h3>
                <p class='category-msg'>{msg}</p>
            </div>
            """)

    if not detected_cards:
        return """
        <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 40px; border-radius: 12px;'>
            <img src="https://clipart-library.com/images/8cA6Ga9Ki.jpg" alt="Good" width='200' height='200' style='border-radius: 20px; box-shadow: 0 0 10px gray;'>
            <h2 style='font-size: 30px;'>This comment looks safe and respectful!</h2>
            <p style='font-size: 18px;'>Thanks for spreading positivity online </p>
        </div>
        """

    return f"""
    <style>
        .card-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }}
        .card {{
            width: 260px;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .card img {{
            width: 200px;
            height: 200px;
            object-fit: cover;
            border-radius: 12px;
            margin-bottom: 10px;
        }}
        .category-title {{
            color: #cc0000 !important;
            font-weight: bold;
            font-size: 30px !important;
            margin-bottom: 10px;
        }}
        .category-msg {{
            font-size: 18px;
            font-family: Arial, sans-serif;
        }}
    </style>
    <div class='card-container'>
        {''.join(detected_cards)}
    </div>
    """

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("<div style='text-align:center; color:#cc0000'><h1>Toxicity Detection Game</h1></div>")
    gr.Markdown("<div style='text-align:center;'><h3>Type a comment and see how toxic it is — and how it emotionally impacts others.</h3></div>")

    output_html = gr.HTML()
    user_input = gr.Textbox(lines=4, placeholder="Write your comment here...", label="Your Comment")
    analyze_button = gr.Button("Analyze Comment")

    analyze_button.click(classify_and_advise, inputs=user_input, outputs=output_html)

demo.launch()
