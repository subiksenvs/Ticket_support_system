# export_ticket_texts.py
import pandas as pd
import pickle

CSV_PATH = "tickets_with_solutions.csv"  # update if your CSV path differs

df = pd.read_csv(CSV_PATH)
# choose the column that contains ticket text; your CSV uses "ticket_text"
ticket_texts = df["ticket_text"].astype(str).tolist()

with open("ticket_texts.pkl", "wb") as f:
    pickle.dump(ticket_texts, f)

print(f"Exported {len(ticket_texts)} ticket texts -> ticket_texts.pkl")
