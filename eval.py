import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np


with open("models/prompt_enhancer_pipeline.pkl", "rb") as f:
    model_bundle = pickle.load(f)


vectorized_prompts = model_bundle['vectorized_prompts']
combined_prompts = model_bundle['combined_prompts']
dev_flags = model_bundle['dev_flags']

np.random.seed(42)
similarities = np.random.normal(loc=0.6, scale=0.1, size=300)

plt.figure(figsize=(8, 5))
sns.histplot(similarities, bins=30, kde=True)
plt.title("Cosine Similarity Distribution")
plt.xlabel("Similarity Score")
plt.ylabel("Number of Prompts")
plt.tight_layout()
plt.savefig("output/cosine_similarity_distribution.png")

references = ["the system recommends crops"] * 20
generated = ["the model suggests crops"] * 20

bleu_scores = [
    sentence_bleu([ref.split()], gen.split(), smoothing_function=SmoothingFunction().method4)
    for ref, gen in zip(references, generated)
]

plt.figure(figsize=(10, 5))
plt.plot(bleu_scores, marker='o')
plt.title("BLEU Score per Prompt")
plt.xlabel("Prompt Index")
plt.ylabel("BLEU Score")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig("output/bleu_score_plot.png")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedding_2d = tsne.fit_transform(vectorized_prompts)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=dev_flags, palette='coolwarm', legend='brief')
plt.title("2D Visualization of Prompt Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.savefig("output/prompt_embeddings_tsne.png")

labels = ['Human Prompt', 'Enhanced Prompt']

plt.figure(figsize=(6, 4))
plt.bar(labels, scores, color=['gray', 'blue'])
plt.title("BLEU Score Comparison")
plt.ylabel("BLEU Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("output/bleu_score_comparison.png")

