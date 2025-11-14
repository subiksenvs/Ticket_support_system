import streamlit as st
import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from pathlib import Path
import os, requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------- Config ----------
API_BASE = "http://127.0.0.1:5000"
st.set_page_config(page_title="🎫 Ticket Uploader & Analyzer", layout="wide")


SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
if not SLACK_WEBHOOK_URL:
    print("❌ Slack webhook is missing. Check your .env file.")
else:
    data = {"text": "Slack integration successful!"}
    response = requests.post(SLACK_WEBHOOK_URL, json=data)
    print(response.status_code, response.text)
# ---------- UI ----------
st.title("🎫 Ticket Upload & Analysis")

# --- Sidebar filters/search ---
st.sidebar.header("Filters & Search")
search_query = st.sidebar.text_input("Search tickets (keyword)")
priority_filter = st.sidebar.selectbox("Priority filter", ["All", "High", "Medium", "Low"])
search_button = st.sidebar.button("🔎 Search Tickets")

# --- File uploader ---
st.subheader("1️⃣ Upload ticket (then press Upload button)")
uploaded_file = st.file_uploader("Choose a .txt or .csv file", type=["txt", "csv"])

# --- Preview ---
preview_content = ""
if uploaded_file:
    uploaded_file.seek(0)
    if uploaded_file.name.lower().endswith(".txt"):
        preview_content = uploaded_file.read().decode("utf-8")
        st.text_area("Ticket preview", preview_content, height=200)
    else:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(StringIO(uploaded_file.read().decode("utf-8")))
            st.dataframe(df)
            # Automatically set ticket_text for recommendation box
            if "content" in df.columns:
                preview_content = str(df.iloc[0]["content"])
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# --- Upload button ---
if st.button("📤 Upload Ticket"):
    if not uploaded_file:
        st.warning("Please choose a file first.")
    else:
        try:
            uploaded_df = pd.DataFrame()  # to store uploaded tickets locally
            
            if uploaded_file.name.lower().endswith(".csv"):
                uploaded_file.seek(0)
                df = pd.read_csv(StringIO(uploaded_file.read().decode("utf-8")))
                
                # Allow either 'content' or 'ticket_text' column
                if "content" not in df.columns and "ticket_text" not in df.columns:
                    st.error("CSV must contain a 'content' or 'ticket_text' column to upload rows.")
                else:
                    column_name = "content" if "content" in df.columns else "ticket_text"
                    successes, failures = 0, 0
                    for _, row in df.iterrows():
                        content = str(row[column_name]).strip()
                        if not content:
                            failures += 1
                            continue
                        payload = {"filename": uploaded_file.name, "content": content}
                        resp = requests.post(f"{API_BASE}/upload", json=payload, timeout=30)
                        if resp.ok:
                            successes += 1
                            uploaded_df = pd.concat([uploaded_df, pd.DataFrame([{"ticket_text": content}])], ignore_index=True)
                        else:
                            failures += 1
                    
                    st.success(f"✅ Uploaded {successes} rows. ❌ Failed: {failures}.")
            
            else:  # txt file
                uploaded_file.seek(0)
                content = uploaded_file.read().decode("utf-8").strip()
                if not content:
                    st.warning("File is empty.")
                else:
                    payload = {"filename": uploaded_file.name, "content": content}
                    resp = requests.post(f"{API_BASE}/upload", json=payload, timeout=30)
                    if resp.ok:
                        st.success(f"✅ Ticket '{uploaded_file.name}' saved successfully!")
                        uploaded_df = pd.DataFrame([{"ticket_text": content}])
                    else:
                        st.error(f"Upload failed: {resp.text}")

            # Update session state for analytics and recommendation
            if not uploaded_df.empty:
                if "df_tickets" not in st.session_state:
                    st.session_state["df_tickets"] = uploaded_df
                else:
                    st.session_state["df_tickets"] = pd.concat(
                        [st.session_state["df_tickets"], uploaded_df],
                        ignore_index=True
                    )
        except Exception as e:
            st.error(f"Error uploading ticket(s): {e}")


# --- Analyze ticket text manually ---
st.subheader("2️⃣ Analyze a ticket text")
ticket_text = st.text_area(
    "Paste ticket text here (or leave empty to use uploaded preview):",
    value=preview_content,
    height=200
)
if st.button("🔍 Analyze Text (Predict Priority)"):
    if not ticket_text.strip():
        st.warning("Add some text to analyze.")
    else:
        try:
            resp = requests.post(f"{API_BASE}/analyze", json={"content": ticket_text.strip()}, timeout=30)
            if resp.ok:
                d = resp.json()
                st.success(f"Predicted Priority: {d.get('priority', 'Unknown')}")
                st.json(d)
            else:
                st.error(f"Analysis failed: {resp.text}")
        except Exception as e:
            st.error(f"Error analyzing ticket: {e}")

st.markdown("---")

# --- Browse stored tickets ---
if search_button:
    try:
        params = {"search": search_query.strip(), "priority": priority_filter}
        resp = requests.get(f"{API_BASE}/tickets", params=params, timeout=30)
        if resp.ok:
            tickets = resp.json()
            if tickets:
                df = pd.DataFrame(tickets)
                st.dataframe(df, use_container_width=True)
                st.write(f"Total: {len(df)} tickets")
                if "priority" in df.columns:
                    st.bar_chart(df["priority"].value_counts())
            else:
                st.info("No tickets matched your criteria.")
        else:
            st.error(f"Failed to fetch tickets: {resp.text}")
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")

st.markdown("---")
st.title("🎯 Ticket Solution Recommendation System")

if "uploaded_ticket_text" not in st.session_state:
    st.session_state["uploaded_ticket_text"] = preview_content

# --- Analyze button ---
ticket_text = st.text_area(
    "Enter your ticket description (or leave empty to use uploaded ticket):",
    value=st.session_state.get("uploaded_ticket_text", ""),
    height=200,
    key="ticket_text_area"
)
if st.button("🔍 Recommend Solution"):
    if not ticket_text.strip():
        st.warning("⚠️ Please enter a ticket description first.")
    else:
        try:
            with st.spinner("🔎 Finding best-matching solution and detecting content gap..."):
                response = requests.post(f"{API_BASE}/recommend", json={"ticket_text": ticket_text})

            if response.status_code == 200:
                data = response.json()
                st.session_state["recommendation"] = data
            else:
                st.error(f"Backend returned {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error(f"⚠️ Error contacting backend: {e}")

# --- Display recommendation if available ---
if "recommendation" in st.session_state:
    data = st.session_state["recommendation"]
    ticket_text = data.get("uploaded_ticket_text", "") 
    if 'recommended_solution' in data:
        st.subheader("🧠 Recommendation Result")
        st.caption(f"Similarity Score: {data.get('similarity_score', 'N/A')}")
        st.write(f"**Gap Status:** {data.get('gap_status', 'N/A')}")

        if not data.get("content_gap", False):
            st.success("✅ Covered by existing knowledge base.")
            st.write("**Recommended Solution:**")
            st.info(data['recommended_solution'])
        else:
            st.warning("🚨 Knowledge Gap Detected!")
            st.info("No similar issue found in the database.")

            if st.button("✨ Generate New Solution"):
                with st.spinner("Generating new solution..."):
                    gen_response = requests.post(
                        f"{API_BASE}/generate_new_solution",
                        json={"ticket_text": ticket_text}
                    )
                if gen_response.status_code == 200:
                    result = gen_response.json()
                    st.session_state["new_solution"] = result["new_solution"]
                    st.success("✅ New Solution Generated Successfully!")
                    st.info(result["new_solution"])
                else:
                    st.error("❌ Failed to generate new solution.")









# ---------- Analytics Dashboard ----------
st.markdown("---")
st.title("📊 Ticket Analytics Dashboard")

if st.button("📈 Show Analytics"):
    # Fetch tickets
    try:
        resp = requests.get(f"{API_BASE}/tickets", params={"search": "", "priority": "All"}, timeout=30)
        if resp.ok:
            df_analytics = pd.DataFrame(resp.json())
        else:
            st.error(f"Failed to fetch tickets: {resp.text}")
            df_analytics = pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")
        df_analytics = pd.DataFrame()

    if not df_analytics.empty:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")

        st.subheader("1️⃣ Tickets per Priority")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="priority", data=df_analytics, order=["High","Medium","Low"], ax=ax1)
        st.pyplot(fig1)

        st.subheader("2️⃣ Ticket Length Distribution")
        df_analytics["word_count"] = df_analytics["content"].apply(lambda x: len(str(x).split()))
        fig2, ax2 = plt.subplots()
        sns.histplot(df_analytics["word_count"], bins=20, kde=True, ax=ax2)
        ax2.set_xlabel("Word Count")
        st.pyplot(fig2)

        # Example: Heatmap for content gap / priority co-occurrence
        if "priority" in df_analytics.columns:
            st.subheader("3️⃣ Priority vs Ticket Category Heatmap")
            # Dummy category column if you have one
            if "category" not in df_analytics.columns:
                df_analytics["category"] = "General"
            heat_data = pd.crosstab(df_analytics["category"], df_analytics["priority"])
            fig3, ax3 = plt.subplots()
            sns.heatmap(heat_data, annot=True, fmt="d", cmap="YlGnBu", ax=ax3)
            st.pyplot(fig3)

        # Example: Line plot (ticket trends over time)
        if "timestamp" in df_analytics.columns:
            st.subheader("4️⃣ Tickets Over Time")
            df_analytics["timestamp"] = pd.to_datetime(df_analytics["timestamp"])
            time_data = df_analytics.groupby(df_analytics["timestamp"].dt.date).size().reset_index(name="count")
            fig4, ax4 = plt.subplots()
            sns.lineplot(x="timestamp", y="count", data=time_data, ax=ax4)
            st.pyplot(fig4)

        # Example: Scatter plot (word count vs priority)
        st.subheader("5️⃣ Ticket Word Count vs Priority")
        fig5, ax5 = plt.subplots()
        sns.scatterplot(x="word_count", y="priority", data=df_analytics, ax=ax5)
        st.pyplot(fig5)

        # Example: Subplots for comparative view
        st.subheader("6️⃣ Comparative Subplots by Priority")
        fig6, axs6 = plt.subplots(1,3, figsize=(15,4))
        priorities = ["High","Medium","Low"]
        for i, p in enumerate(priorities):
            sns.histplot(df_analytics[df_analytics["priority"]==p]["word_count"], bins=10, ax=axs6[i])
            axs6[i].set_title(f"{p} Priority")
        st.pyplot(fig6)

    else:
        st.info("No tickets available for analytics.")



# --- Function to display content-gap tickets ---
def show_content_gap_tickets():
    """Fetch and display content-gap tickets from backend API."""
    try:
        resp = requests.get(f"{API_BASE}/content-gap-tickets", timeout=30)
        if resp.ok:
            df_gap = pd.DataFrame(resp.json())
            if not df_gap.empty:
                st.dataframe(df_gap, use_container_width=True)
                st.write(f"Total tickets with content gaps: {len(df_gap)}")
            else:
                st.info("✅ No content gap tickets found.")
        else:
            st.error(f"Failed to fetch content-gap tickets: {resp.text}")
    except Exception as e:
        st.error(f"Error fetching content-gap tickets: {e}")

        
# --- Display content-gap tickets section ---
st.markdown("---")
st.title("📌 Content Gap Tickets")

# Manual refresh button always re-reads CSV
if st.button("Refresh Content Gap Tickets"):
    show_content_gap_tickets()
else:
    # Show tickets automatically on page load
    show_content_gap_tickets()


st.title(" Update Ticket Solution (Admin Only)")

st.markdown("""
Paste the ticket text for which a content gap was detected and enter the new solution manually.
This will update your CSV dataset and embeddings.
""")



# --- Initialize session state for clearing ---
if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = ""
if "new_solution" not in st.session_state:
    st.session_state.new_solution = ""
if "password" not in st.session_state:
    st.session_state.password = ""

# --- Form for updating solution ---
with st.form("update_solution_form"):
    ticket_text = st.text_area("Ticket Text", height=200,key="ticket_text")
    new_solution = st.text_area("New Solution", height=150, key="new_solution")
    password = st.text_input("Admin Password", type="password", key="password")
    
    # Submit button inside form
    submitted = st.form_submit_button("💾 Update Solution")

if submitted:
    if not ticket_text.strip() or not new_solution.strip():
        st.warning("Please provide both ticket text and the new solution.")
    elif not password.strip():
        st.warning("Please enter admin password.")
    else:
        try:
            payload = {
                "ticket_text": ticket_text.strip(),
                "new_solution": new_solution.strip(),
                "password": password.strip()
            }
            response = requests.post(f"{API_BASE}/update_solution_manual", json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                st.success(data.get("message", "✅ Solution updated successfully!"))
                st.info(f"✅ Updated Solution:\n{data.get('solution')}")
                # --- Clear text areas after successful update --
                st.session_state.ticket_text = ""
                st.session_state.new_solution = ""
                st.session_state.password = ""



            elif response.status_code == 403:
                st.error("❌ Invalid admin password.")
            else:
                st.error(f"Failed to update solution. Status: {response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error(f"Error contacting backend: {e}")



GAP_CSV_PATH = Path(r"C:/Users/POOJITHA/Documents/Knowledge Engine/backend/content_gap_tickets.csv")

# --- Load CSV ---
df_gap = pd.read_csv(GAP_CSV_PATH, encoding="utf-8-sig")

st.title(" Content Gap Ticket Manager")

# Ensure file exists
if not GAP_CSV_PATH.exists():
    pd.DataFrame(columns=["ticket_id","ticket_text","similarity_score","status","created_at"]).to_csv(GAP_CSV_PATH, index=False, encoding="utf-8-sig")

df_gap = pd.read_csv(GAP_CSV_PATH, encoding="utf-8-sig")
pending = df_gap[df_gap["status"].str.lower() == "pending"]

if pending.empty:
    st.success("🎉 No pending tickets.")
else:
    for idx, row in pending.iterrows():
        ticket_id = row.get("ticket_id", "")
        ticket_text = row.get("ticket_text", "")
        similarity = row.get("similarity_score", "")
        created_at = row.get("created_at", "")

        st.markdown(f"**Ticket ID:** {ticket_id}  \n**Content:** {ticket_text}  \n**Similarity:** {similarity}  \n**Created:** {created_at}")
        

        
        if st.button(f"✅ Mark as Solved - {ticket_id}", key=f"solve_{idx}"):
            
            solved_at = datetime.utcnow().isoformat() + "Z"
            df_gap.loc[idx, "status"] = "Solved"
            df_gap.loc[idx, "solved_by"] = admin_name or "Admin"
            df_gap.loc[idx, "solved_at"] = solved_at
            df_gap.to_csv(GAP_CSV_PATH, index=False, encoding="utf-8-sig")
            st.write("---")

            # Send Slack notification (rich)
            if SLACK_WEBHOOK_URL:
                slack_payload = {
                    "blocks": [
                        {"type":"section","text":{"type":"mrkdwn","text":":white_check_mark: *Ticket Solved!*"}},
                        {"type":"section","fields":[
                            {"type":"mrkdwn","text":f"*Ticket ID:*\n{ticket_id}"},
                            {"type":"mrkdwn","text":f"*Solved by:*\n{admin_name or 'Admin'}"},
                            {"type":"mrkdwn","text":f"*Solved at (UTC):*\n{solved_at}"},
                            {"type":"mrkdwn","text":f"*Similarity:*\n{similarity}"}
                        ]},
                        {"type":"section","text":{"type":"mrkdwn","text":f"*Content:*\n>{ticket_text}"}}
                    ]
                }
                try:
                    resp = requests.post(SLACK_WEBHOOK_URL, json=slack_payload, timeout=5)
                    if resp.ok:
                        st.info("📢 Slack notified.")
                    else:
                        st.warning(f"⚠️ Slack webhook returned: {resp.status_code}")
                except Exception as e:
                    st.error(f"⚠️ Slack request failed: {e}")

            # Refresh view (Streamlit >=1.30)
            st.rerun()
