from datasets import load_dataset
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "tickets.db"

ds = load_dataset("Tobi-Bueck/customer-support-tickets")["train"]

conn = sqlite3.connect(DB_PATH)
for rec in ds:
    content = rec["body"]
    priority = rec.get("priority", "Low").capitalize()
    conn.execute(
        "INSERT INTO tickets (filename, content, priority) VALUES (?, ?, ?)",
        ("dataset_ticket", content, priority)
    )
conn.commit()
conn.close()
print("✅ Dataset loaded into tickets.db")
