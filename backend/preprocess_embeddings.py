# preprocess_embeddings.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

print("Loading model and CSV...")
df = pd.read_csv("tickets_with_solutions.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding all ticket texts...")
embeddings = model.encode(df["ticket_text"].tolist(), convert_to_tensor=True)
embeddings = embeddings.cpu()

print("Saving embeddings...")
with open("ticket_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Embeddings saved to ticket_embeddings.pkl")
