import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/prompts.csv")

df['combined'] = df['act'] + ": " + df['prompt']
df['for_devs'] = df['for_devs'].astype(bool).astype(int)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 2)
    )),
    ("svd", TruncatedSVD(n_components=100, random_state=42))
])

vectorized_features = pipeline.fit_transform(df['combined'])

model_bundle = {
    "pipeline": pipeline,
    "vectorized_prompts": vectorized_features,
    "combined_prompts": df['combined'].tolist(),
    "dev_flags": df['for_devs'].tolist(),
    "raw_df": df.to_dict(orient="records")  
}

with open("models/prompt_enhancer_pipeline.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("Vectorizer pipeline saved to models/prompt_enhancer_pipeline.pkl")
