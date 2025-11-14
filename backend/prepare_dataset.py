from datasets import load_dataset
import pandas as pd

# Load dataset from Hugging Face
dataset = load_dataset("Tobi-Bueck/customer-support-tickets")

# Convert to DataFrame
df = pd.DataFrame(dataset['train'])

# Inspect the columns
print(df.columns)

# Keep only ticket text and resolution/solution columns
# Adjust based on actual column names (could be 'ticket', 'resolution', 'solution', 'answer', etc.)
df_clean = df[['body', 'answer']].rename(columns={'body': 'ticket_text', 'answer': 'solution'})

# Drop missing entries
df_clean = df_clean.dropna()

# Save as CSV
df_clean.to_csv("tickets_with_solutions.csv", index=False)

print("✅ tickets_with_solutions.csv created successfully!")
