# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3, pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import random
from datetime import datetime
import os
import requests
import html
import json
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
# # ---------- Config ----------
DB_PATH = Path(__file__).parent / "tickets.db"
CSV_PATH = "tickets_with_solutions.csv"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TRAINED_MODEL_PATH = Path(__file__).parent / "model" / "checkpoint-5807"
EMBEDDING_FILE = "ticket_embeddings.pkl"

# ---------- Initialize Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Load Models ----------
print("🔹 Loading models...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ---------- Load Dataset ----------
print("🔹 Loading dataset...")
df_tickets = pd.read_csv(CSV_PATH)

# ---------- Load or Precompute Embeddings ----------
if Path(EMBEDDING_FILE).exists():
    print("🔹 Loading precomputed embeddings from file...")

    # Force loading on CPU (even if file was saved from GPU)
    with open(EMBEDDING_FILE, "rb") as f:
        try:
            ticket_embeddings = torch.load(f, map_location=torch.device('cpu'))
        except Exception:
            # Fallback for non-torch pickle data
            f.seek(0)
            ticket_embeddings = pickle.load(f)

    print("✅ Embeddings loaded successfully.")

else:
    print("⚙️ Encoding all ticket texts (first-time only)...")
    ticket_embeddings = embedding_model.encode(
        df_tickets["ticket_text"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    )

    # Save embeddings in CPU format for portability
    with open(EMBEDDING_FILE, "wb") as f:
        torch.save(ticket_embeddings.cpu(), f)

    print("✅ Embeddings saved for future use!")

# ---------- Load Fine-tuned Classification Model ----------
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
classification_model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODEL_PATH)

# ---------- Database Setup ----------
def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content TEXT,
            priority TEXT,
            embedding BLOB
        );
        """)
        conn.commit()

def insert_ticket_to_db(filename, content, priority, embedding):
    emb_blob = pickle.dumps(embedding)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO tickets (filename, content, priority, embedding) VALUES (?, ?, ?, ?)",
            (filename, content, priority, emb_blob)
        )
        conn.commit()

def fetch_tickets_from_db(search="", priority_filter="All"):
    query = "SELECT id, filename, content, priority FROM tickets WHERE 1=1"
    params = []
    if search:
        query += " AND content LIKE ?"
        params.append(f"%{search}%")
    if priority_filter != "All":
        query += " AND priority = ?"
        params.append(priority_filter)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()
    return [{"id":r[0], "filename":r[1], "content":r[2], "priority":r[3]} for r in rows]

# ---------- Priority Prediction ----------
def predict_priority(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = classification_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label_id = torch.argmax(probs).item()
    labels = ["Low", "Medium", "High"]
    return labels[label_id] if label_id < len(labels) else "Unknown"


# ---------- Content Gap Detection ----------
def detect_knowledge_gap(ticket_text):
    """
    Detects if a new ticket is not similar to any existing ticket in the dataset.
    Uses cosine similarity between the new ticket and precomputed ticket embeddings.
    """
    query_emb = embedding_model.encode(ticket_text, convert_to_tensor=True)
    similarities = util.cos_sim(query_emb, ticket_embeddings).squeeze()

    max_similarity = float(torch.max(similarities))

    # You can adjust this threshold (0.5 is a good starting point)
    if max_similarity < 0.5:
        gap_status = "Knowledge Gap Detected"
    else:
        gap_status = "Covered by Existing Knowledge"

    return {
        "gap_status": gap_status,
        "max_similarity": round(max_similarity, 3)
    }

# ---------- Routes ----------
@app.route("/upload", methods=["POST"])
def upload_ticket():
    try:
        if request.is_json:
            data = request.get_json()
            filename = data.get("filename", "unknown")
            content = data.get("content", "").strip()
            
        elif "file" in request.files:
            f = request.files["file"]
            filename = f.filename
            content = f.read().decode("utf-8").strip()
        else:
            return jsonify({"error": "No file or JSON provided"}), 400

        if not content:
            return jsonify({"error": "Empty content"}), 400

        priority = predict_priority(content)
        embedding = embedding_model.encode(content)

        insert_ticket_to_db(filename, content, priority, embedding)

        return jsonify({
            "message": "Ticket saved",
            "filename": filename,
            "priority": priority,
            "embedding_dim": len(embedding)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze_ticket():
    data = request.get_json(force=True)
    text = data.get("content", "")
    if not text:
        return jsonify({"error": "content required"}), 400
    priority = predict_priority(text)
    embedding = embedding_model.encode(text)
    return jsonify({"priority": priority, "embedding_dim": len(embedding)}), 200


@app.route("/tickets", methods=["GET"])
def list_tickets():
    search = request.args.get("search", "")
    priority = request.args.get("priority", "All")
    return jsonify(fetch_tickets_from_db(search, priority)), 200

GAP_CSV_PATH = Path(__file__).parent / "content_gap_tickets.csv"
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
if SLACK_WEBHOOK_URL:
    requests.post(SLACK_WEBHOOK_URL, json={"text": "Backend alert!"})
else:
    print("❌ Slack webhook missing. Add it to your .env file.")
def send_slack_notification(ticket_id, ticket_text, similarity_score, timestamp, webhook_url, solved=False):
    if not webhook_url:
        print("⚠️ Slack webhook URL not set.")
        return

    max_length = 2000
    safe_text = html.escape(ticket_text)
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length] + "…"

    if solved:
        title_text = ":white_check_mark: *Content Gap Ticket Solved*"
        status_text = "Solved"
    else:
        title_text = ":warning: *New Content Gap Ticket Created*"
        status_text = "Pending"

    payload = {
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": title_text}},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Ticket ID:*\n{ticket_id}"},
                {"type": "mrkdwn", "text": f"*Similarity:*\n{similarity_score:.2f}"},
                {"type": "mrkdwn", "text": f"*Status:*\n{status_text}"},
                {"type": "mrkdwn", "text": f"*Updated:*\n{timestamp}"}
            ]},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Content:*\n>{safe_text}"}}
        ]
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        print("Slack HTTP status:", resp.status_code, resp.text)
        if resp.ok:
            print("✅ Slack notification sent successfully.")
        else:
            print("⚠️ Slack notification failed:", resp.status_code, resp.text)
    except Exception as e:
        print("⚠️ Slack notification failed:", e)


# ---------------- Save Content Gap Ticket Function ----------------
def save_content_gap_ticket(ticket_text, similarity_score):
    ticket_text = ticket_text.strip()
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Load or create CSV
    if GAP_CSV_PATH.exists():
        df_gap = pd.read_csv(GAP_CSV_PATH, encoding="utf-8-sig")
        df_gap["ticket_text"] = df_gap["ticket_text"].astype(str).str.strip()
    else:
        df_gap = pd.DataFrame(columns=["ticket_id","ticket_text", "similarity_score", "status", "created_at"])

    # Check duplicates
    ticket_exists = df_gap["ticket_text"].str.lower().eq(ticket_text.lower()).any()
    if not ticket_exists:
        ticket_id = int(datetime.utcnow().timestamp())
        new_row = {
            "ticket_id": ticket_id,
            "ticket_text": ticket_text,
            "similarity_score": similarity_score,
            "status": "Pending",
            "created_at": timestamp
        }
        df_gap = pd.concat([df_gap, pd.DataFrame([new_row])], ignore_index=True)
        df_gap.to_csv(GAP_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"✅ Added content-gap ticket: {ticket_text}")
    else:
        ticket_id = int(df_gap.loc[df_gap["ticket_text"].str.lower() == ticket_text.lower(), "ticket_id"].values[0])
        print(f"ℹ️ Ticket already exists in CSV: {ticket_text}")

    # Send Slack notification
    print("📤 Sending Slack notification for ticket:", ticket_text)
    send_slack_notification(ticket_id, ticket_text, similarity_score, timestamp, SLACK_WEBHOOK_URL)






@app.route("/recommend", methods=["POST"])
def recommend_solution():
    """Recommend a solution, detect content gap, and trigger new solution generation if needed."""
    try:
        ticket_text = ""

        # --- Handle JSON input ---
        if request.is_json:
            data = request.get_json()
            ticket_text = data.get("ticket_text", "").strip()

        # --- Handle uploaded file input ---
        if not ticket_text and request.files:
            uploaded_file = next(iter(request.files.values()))
            
            file_content = uploaded_file.read().decode("utf-8", errors="ignore")
            ticket_text = file_content.strip()

        # --- Validate ---
        if not ticket_text:
            return jsonify({"error": "No ticket text provided"}), 400

        # --- Encode input ticket ---
        query_emb = embedding_model.encode(ticket_text, convert_to_tensor=True)

        # --- Compute similarity scores ---
        similarities = util.cos_sim(query_emb, ticket_embeddings).squeeze()

        # --- Get best match ---
        best_idx = torch.argmax(similarities).item()
        top_match = df_tickets.iloc[best_idx]
        max_similarity = float(similarities[best_idx])

        # --- Clean placeholders ---
        recommended_solution = str(top_match["solution"])
        recommended_solution = (
            recommended_solution
            .replace("<name>", "customer")
            .replace("<NAME>", "customer")
            .replace("<tel_num>", "your contact number")
            .replace("<acc_num>", "your account number")
        )

        # --- Detect Knowledge Gap ---
        gap_threshold = 0.25
        content_gap = max_similarity < gap_threshold

        # --- Save ticket if content gap detected ---
        if content_gap:
            save_content_gap_ticket(ticket_text, max_similarity)

        # --- Prepare Response ---
        response = {
            "uploaded_ticket_text": ticket_text,
            "recommended_solution": recommended_solution,
            "similarity_score": round(max_similarity, 3),
            "gap_status": "Knowledge Gap Detected" if content_gap else "Covered by Existing Knowledge",
            "content_gap": content_gap,
            "show_generate_button": content_gap
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/detect-gap", methods=["POST"])
def detect_gap():
    """
    API endpoint for detecting missing or outdated knowledge areas.
    """
    try:
        data = request.get_json()
        ticket_text = data.get("ticket_text", "").strip()

        if not ticket_text:
            return jsonify({"error": "No ticket text provided"}), 400

        result = detect_knowledge_gap(ticket_text)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/add-solution", methods=["POST"])
def add_new_solution():
    """
    Adds a new solution to the dataset automatically or via user input.
    """
    try:
        data = request.get_json()
        ticket_text = data.get("ticket_text", "").strip()
        new_solution = data.get("solution", "")

        if not ticket_text:
            return jsonify({"error": "No ticket text provided"}), 400

        # If no manual solution provided, auto-generate a simple placeholder
        if not new_solution:
            new_solution = random.choice([
                "Issue acknowledged, our team is investigating.",
                "Reset configuration and retry after clearing cache.",
                "Escalate this issue to Tier 2 support for deeper investigation.",
                "Perform a clean reinstallation of the application.",
                "Restart the system and check network connectivity before retrying."
            ])

        # Append to the dataset CSV
        global df_tickets, ticket_embeddings
        new_entry = pd.DataFrame([{"ticket_text": ticket_text, "solution": new_solution}])
        df_tickets = pd.concat([df_tickets, new_entry], ignore_index=True)
        df_tickets.to_csv(CSV_PATH, index=False)

        # Update embeddings dynamically
        new_emb = embedding_model.encode([ticket_text], convert_to_tensor=True)
        ticket_embeddings = torch.cat((ticket_embeddings, new_emb), dim=0)
        with open(EMBEDDING_FILE, "wb") as f:
            torch.save(ticket_embeddings.cpu(), f)

        return jsonify({
            "message": "✅ New solution added successfully!",
            "solution": new_solution
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/generate_new_solution", methods=["POST"])
def generate_new_solution():
    """Generate a new solution automatically when a content gap is detected."""
    data = request.get_json()
    ticket_text = data.get("ticket_text", "").strip()

    if not ticket_text:
        return jsonify({"error": "No ticket text provided"}), 400

    try:
        # Example: simple template-based generation (you can replace this with your model)
        new_solution = (
            f"Dear customer, thank you for reaching out regarding the issue: '{ticket_text}'. "
            "We are sorry for the inconvenience caused. Our team will review your request and "
            "provide a suitable resolution shortly. Please ensure you have shared your account "
            "details or order reference for faster support. Thank you for your patience."
        )

        # Optionally save this generated solution
        with open("generated_solutions.csv", "a", encoding="utf-8") as f:
            f.write(f"{ticket_text},{new_solution}\n")

        return jsonify({
            "message": "New solution generated successfully.",
            "new_solution": new_solution
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/update_solution_manual", methods=["POST"])
def update_solution_manual():
    """
    Manually update the solution for a content-gap ticket.
    JSON body should contain: {"ticket_text": "...", "new_solution": "...", "password": "..."}
    """
    try:
        data = request.get_json()
        ticket_text = data.get("ticket_text", "").strip()
        new_solution = data.get("new_solution", "").strip()
        password = data.get("password", "").strip()

        # --- Password check ---
        PREDEFINED_PASSWORD = "mypassword123"  # same as frontend
        if password != PREDEFINED_PASSWORD:
            return jsonify({"error": "Invalid admin password"}), 403

        if not ticket_text or not new_solution:
            return jsonify({"error": "ticket_text and new_solution are required"}), 400

        # --- Update or append solution in main dataset ---
        global df_tickets, ticket_embeddings
        existing_idx = df_tickets.index[df_tickets["ticket_text"].str.strip().str.lower() == ticket_text.lower()].tolist()
        if existing_idx:
            df_tickets.at[existing_idx[0], "solution"] = new_solution
        else:
            new_entry = pd.DataFrame([{"ticket_text": ticket_text, "solution": new_solution}])
            df_tickets = pd.concat([df_tickets, new_entry], ignore_index=True)
            # Update embeddings
            new_emb = embedding_model.encode([ticket_text], convert_to_tensor=True)
            ticket_embeddings = torch.cat((ticket_embeddings, new_emb), dim=0)

        # --- Save main dataset ---
        df_tickets.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        with open(EMBEDDING_FILE, "wb") as f:
            torch.save(ticket_embeddings.cpu(), f)

        # --- Update content-gap CSV to mark ticket as Solved ---
        GAP_CSV_PATH = Path(__file__).parent / "content_gap_tickets.csv"
        if GAP_CSV_PATH.exists():
            df_gap = pd.read_csv(GAP_CSV_PATH, encoding="utf-8-sig")

    # Ensure 'status' column exists
        if "status" not in df_gap.columns:
            df_gap["status"] = "Pending"

    # Normalize ticket text for comparison
        ticket_text_norm = ticket_text.lower().strip()
        df_gap["ticket_text_norm"] = df_gap["ticket_text"].astype(str).str.lower().str.strip()

    # Find matching row
        mask = df_gap["ticket_text_norm"] == ticket_text_norm
        if mask.any():
            df_gap.loc[mask, "status"] = "Solved"
            ticket_id = int(df_gap.loc[mask, "ticket_id"].values[0])  # get ticket_id for Slack

        # Drop helper column and save
            df_gap = df_gap.drop(columns=["ticket_text_norm"])
            df_gap.to_csv(GAP_CSV_PATH, index=False, encoding="utf-8-sig")

        # Send Slack notification for solved ticket
            timestamp = datetime.utcnow().isoformat() + "Z"
            send_slack_notification(
                ticket_id=ticket_id,
                ticket_text=ticket_text,
                similarity_score=0,
                timestamp=timestamp,
                webhook_url=SLACK_WEBHOOK_URL,
                solved=True  # You can add a parameter to show "solved" type
            )

            print(f"✅ Ticket marked as Solved: {ticket_text}")
        else:
            print("⚠️ Ticket not found in content gap CSV.")

        return jsonify({
            "message": "✅ Solution updated successfully and ticket marked as Solved!",
            "solution": new_solution
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@app.route("/content-gap-tickets", methods=["GET"])
def get_content_gap_tickets():
    """Return all tickets that have content gaps."""
    try:
        if GAP_CSV_PATH.exists():
            df_gap = pd.read_csv(GAP_CSV_PATH)
            df_gap = df_gap.fillna("").to_dict(orient="records")
            return jsonify(df_gap), 200
        else:
            return jsonify([]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/mark_ticket_solved", methods=["POST"])
def mark_ticket_solved():
    """
    Manually mark a content-gap ticket as 'Solved' via a button click.
    JSON body: {"ticket_text": "...", "password": "..."}
    """
    try:
        data = request.get_json()
        ticket_text = data.get("ticket_text", "").strip()
        password = data.get("password", "").strip()

        PREDEFINED_PASSWORD = "mypassword123"  # same as frontend
        if password != PREDEFINED_PASSWORD:
            return jsonify({"error": "Invalid admin password"}), 403

        if not ticket_text:
            return jsonify({"error": "ticket_text is required"}), 400

        if not GAP_CSV_PATH.exists():
            return jsonify({"error": "content_gap_tickets.csv not found"}), 404

        df_gap = pd.read_csv(GAP_CSV_PATH, encoding="utf-8-sig")
        df_gap["ticket_text_clean"] = df_gap["ticket_text"].astype(str).str.strip().str.lower()
        ticket_text_clean = ticket_text.lower().strip()

        # Match & update
        mask = df_gap["ticket_text_clean"] == ticket_text_clean
        if mask.any():
            df_gap.loc[mask, "status"] = "Solved"
            df_gap.drop(columns=["ticket_text_clean"], inplace=True, errors="ignore")
            df_gap.to_csv(GAP_CSV_PATH, index=False, encoding="utf-8-sig")
            print(f"✅ Ticket marked as Solved: {ticket_text}")
            return jsonify({"message": f"✅ '{ticket_text}' marked as Solved!"}), 200
        else:
            return jsonify({"error": "Ticket not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Run App ----------
print("🚀 Starting Flask app...")
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
