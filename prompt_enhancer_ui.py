import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import random
from llama_util import init_llama
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd

@st.cache_resource
def load_model():
    with open("models/prompt_enhancer_pipeline.pkl", "rb") as f:
        return pickle.load(f)

data = load_model()
pipeline = data['pipeline']
vectors = data['vectorized_prompts']
prompts = data['combined_prompts']

llama_model = init_llama(model_path="/home/vignesh/Desktop/NLP/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")


def generate_enhanced_prompt(model, user_input):
    system_prompt = (
        "You are a helpful assistant that rewrites user input into a better prompt for a language model.\n\n"
        f"User Input: {user_input}\n"
        f"Rewrite this as an improved prompt:"
    )
    output = model(
        system_prompt,
        max_tokens=80,
        temperature=0.7,
        stop=["</s>"]
    )
    enhanced = output["choices"][0]["text"].strip()
    return enhanced if enhanced else "Please give a summary of this text: " + user_input

def enhance_prompt(user_input):
    user_vec = pipeline.transform([user_input])
    similarity = cosine_similarity(user_vec, vectors)[0]
    best_idx = similarity.argmax()
    return prompts[best_idx]

def preprocess_text(text):
    text = text.lower()
    return text

def calculate_bleu(reference, candidate):
    reference = preprocess_text(reference).split()
    candidate = preprocess_text(candidate).split()
    
    return sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))

def get_llama_response(llama_model, prompt):
    output = llama_model(
        prompt,
        max_tokens=60,
        temperature=0.7,
        stop=["</s>"]
    )
    return output["choices"][0]["text"].strip()

st.title("Prompt Enhancer for Chatbots")

st.markdown("""
This tool enhances human-style prompts using TF-IDF + semantic similarity and compares their performance through BLEU scoring.
""")

user_input = st.text_area(" Enter your human-style prompt", height=150)

if user_input:
    enhanced_prompt = enhance_prompt(user_input)
    st.subheader(" Enhanced Prompt")
    st.success(enhanced_prompt)

    if st.button("Send Both to LLaMA"):
        human_response = get_llama_response(llama_model, user_input)
        enhanced_response = get_llama_response(llama_model, enhanced_prompt)

        human_bleu = calculate_bleu(user_input, human_response)
        enhanced_bleu = calculate_bleu(enhanced_prompt, enhanced_response)

        st.markdown("###BLEU Score Comparison")
        st.write(f"**Human Prompt BLEU Score:** `{human_bleU}`")
        st.write(f"**Enhanced Prompt BLEU Score:** `{enhanced_bleU}`")

        bleu_scores = pd.DataFrame(
            {"BLEU Score (%)": [human_bleU, enhanced_bleU]},
            index=["Human", "Enhanced"]
        )
        st.bar_chart(bleu_scores)

st.markdown("---")
st.caption("Built with Streamlit + TF-IDF + LLaMA.")
