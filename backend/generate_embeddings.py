import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------- CONFIG ----------------------
DATA_PATH = "tickets_with_solutions.csv"   # path to your dataset
TEXT_COLUMN = "ticket_text" # column name in your dataset
MODEL_NAME = "intfloat/e5-small-v2"  # fast & accurate
OUTPUT_PATH = "ticket_embeddings.pkl"
# ----------------------------------------------------

# ✅ Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# ✅ Load model
model = SentenceTransformer(MODEL_NAME, device=device)

# ✅ Load dataset
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"Column '{TEXT_COLUMN}' not found in dataset columns: {df.columns}")

ticket_texts = df[TEXT_COLUMN].astype(str).tolist()
print(f"✅ Total tickets to embed: {len(ticket_texts)}")

# ✅ Encode with progress bar
print("⚙️ Generating embeddings...")
embeddings = model.encode(
    ticket_texts,
    batch_size=64,               
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# ✅ Force embeddings to CPU before saving
embeddings = embeddings.cpu()

# ✅ Save embeddings
torch.save(embeddings, OUTPUT_PATH)
print(f"💾 Embeddings saved to {OUTPUT_PATH}")

# ✅ Optional: save a mapping of index → text
df_mapping = pd.DataFrame({
    "id": range(len(ticket_texts)),
    "ticket_text": ticket_texts
})
df_mapping.to_csv("ticket_mapping.csv", index=False)
print("📘 Mapping saved as ticket_mapping.csv")

print("🎉 Embedding generation completed successfully!")
