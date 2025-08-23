import streamlit as st
import json
import uuid
import random
import os
import csv

# Set wide layout FIRST - this is crucial
st.set_page_config(layout="wide")

# --- Custom CSS for styling ---
st.markdown(
    """
    <style>
    .stApp, .stApp * { font-size: 22px !important; }
    
    /* Force text areas to be much wider */
    .stTextArea > div > div > textarea {
        width: 100% !important;
        min-width: 600px !important;
        font-size: 26px !important;
        color: black !important;
    }
    
    .stSlider label { font-size: 26px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Unique widget suffix ---
if "widget_suffix" not in st.session_state:
    st.session_state.widget_suffix = str(uuid.uuid4())[:8]

# --- Load Data ---
@st.cache_data
def load_data():
    with open("Final_Clinical_Reasoning_cleaned.json", "r") as f:
        return json.load(f)

data = load_data()
patient_ids = list(data.keys())

# --- Logging Setup ---
log_file = "evaluation_log.csv"
# Initialize completed_ids set from existing CSV
if os.path.exists(log_file):
    with open(log_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing_rows = list(reader)
        # Skip header
        completed_ids = {row[0] for row in existing_rows[1:]} if len(existing_rows) > 1 else set()
else:
    # Create new file with header
    with open(log_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Patient ID",
            "Standard_Reasoning", "Standard_Relevance", "Standard_Usefulness",
            "RAG_Reasoning", "RAG_Relevance", "RAG_Usefulness"
        ])
    completed_ids = set()

# --- Session State Initialization ---
if "initialized" not in st.session_state:
    # Find first index not in completed_ids
    start_idx = 0
    for i, pid in enumerate(patient_ids):
        if pid not in completed_ids:
            start_idx = i
            break
    else:
        start_idx = 0  # all completed
    
    st.session_state.current_index = start_idx
    st.session_state.order_map = {}
    st.session_state.completed_ids = completed_ids
    st.session_state.initialized = True

# --- Navigation Callbacks ---
def go_previous():
    # Move backwards to previous uncompleted
    idx = st.session_state.current_index
    n = len(patient_ids)
    for _ in range(n):
        idx = (idx - 1) % n
        if patient_ids[idx] not in st.session_state.completed_ids:
            st.session_state.current_index = idx
            return

def go_next():
    idx = st.session_state.current_index
    pid = patient_ids[idx]
    
    # Determine left/right mapping
    order_left_standard = st.session_state.order_map.get(idx, True)
    left_method, right_method = ("standard", "rag") if order_left_standard else ("rag", "standard")
    
    # Retrieve ratings for Option 1
    rel1 = st.session_state.get(f"relevance1_{st.session_state.widget_suffix}")
    r1 = st.session_state.get(f"reasoning1_{st.session_state.widget_suffix}")
    use1 = st.session_state.get(f"usefulness1_{st.session_state.widget_suffix}")
    # Retrieve ratings for Option 2
    rel2 = st.session_state.get(f"relevance2_{st.session_state.widget_suffix}")
    r2 = st.session_state.get(f"reasoning2_{st.session_state.widget_suffix}")
    use2 = st.session_state.get(f"usefulness2_{st.session_state.widget_suffix}")
    
    # Map ratings to Standard and RAG
    if left_method == "standard":
        std_relevance, std_reasoning, std_usefulness = rel1, r1, use1
        rag_relevance, rag_reasoning, rag_usefulness = rel2, r2, use2
    else:
        std_relevance, std_reasoning, std_usefulness = rel2, r2, use2
        rag_relevance, rag_reasoning, rag_usefulness = rel1, r1, use1
    
    # Append to CSV
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            pid,
            std_reasoning, std_relevance, std_usefulness,
            rag_reasoning, rag_relevance, rag_usefulness
        ])
    # Mark completed
    st.session_state.completed_ids.add(pid)
    
    # Advance to next uncompleted
    n = len(patient_ids)
    next_idx = idx
    for _ in range(n):
        next_idx = (next_idx + 1) % n
        if patient_ids[next_idx] not in st.session_state.completed_ids:
            st.session_state.current_index = next_idx
            return
    # If none left, stay on current
    st.session_state.current_index = idx

# --- Guard Empty ---
if not patient_ids:
    st.error("No patient data available.")
    st.stop()

# If all completed, show message
if len(st.session_state.completed_ids) == len(patient_ids):
    st.success("All patients have been evaluated.")
    st.stop()

# --- Current Patient ---
idx = st.session_state.current_index
pid = patient_ids[idx]
patient_data = data[pid]

# --- Assign Random Order if New ---
if idx not in st.session_state.order_map:
    st.session_state.order_map[idx] = random.choice([True, False])
order_left_standard = st.session_state.order_map[idx]

# --- Header ---
st.markdown(f"#### Patient {idx+1}/{len(patient_ids)}: {pid}")

# --- Prompt (Patient Summary) ---
cols = st.columns([1, 10, 1])  # Middle column wider
prompt = patient_data.get("Patient Summaries", "")
cols[1].markdown("**Patient Summary**")
cols[1].text_area(
    "",
    prompt,
    height=200,
    key=f"prompt_{st.session_state.widget_suffix}",
    disabled=True
)

# --- Determine Options ---
left_method, right_method = ("standard", "rag") if order_left_standard else ("rag", "standard")

# --- Definitions for Metrics ---
relevance_def = "_Assesses whether the LLM’s reasoning focuses on clinically pertinent features (e.g., vital signs, imaging, guidelines). Mainly use Section 1's response to evaluate this_"
reasoning_def = "_Assesses the logical coherence and depth of the LLM’s explanation. Mainly use Section 3's response to evaluate this_"
usefulness_def = "_Assesses how actionable and helpful the LLM’s response would be for actual clinical decision-making. (Overall thoughts)_"

# --- Display Responses and Metrics ---
col1, col2 = st.columns([1, 1])  # Equal-width columns

# --- Option 1 (Left) ---
if left_method == "standard":
    resp1 = patient_data.get("Predicted Disposition", "*Not available*")
else:
    resp1 = patient_data.get("RAG Predicted Disposition", "*Not available*")

col1.markdown("**Option 1**")
col1.text_area(
    "",
    resp1,
    height=300,
    key=f"resp1_{st.session_state.widget_suffix}",
    disabled=True
)
# Clinical Relevance Slider
col1.markdown("**Clinical Relevance of Reasoning (1–5)**")
col1.markdown(relevance_def)
col1.slider(
    "",
    1, 5, 3,
    key=f"relevance1_{st.session_state.widget_suffix}"
)
# Reasoning Quality Slider
col1.markdown("**Reasoning Quality (1–5)**")
col1.markdown(reasoning_def)
col1.slider(
    "",
    1, 5, 3,
    key=f"reasoning1_{st.session_state.widget_suffix}"
)
# Practical Usefulness Slider
col1.markdown("**Practical Usefulness (1–5)**")
col1.markdown(usefulness_def)
col1.slider(
    "",
    1, 5, 3,
    key=f"usefulness1_{st.session_state.widget_suffix}"
)

# --- Option 2 (Right) ---
if right_method == "standard":
    resp2 = patient_data.get("Predicted Disposition", "*Not available*")
else:
    resp2 = patient_data.get("RAG Predicted Disposition", "*Not available*")

col2.markdown("**Option 2**")
col2.text_area(
    "",
    resp2,
    height=300,
    key=f"resp2_{st.session_state.widget_suffix}",
    disabled=True
)
# Clinical Relevance Slider
col2.markdown("**Clinical Relevance of Reasoning (1–5)**")
col2.markdown(relevance_def)
col2.slider(
    "",
    1, 5, 3,
    key=f"relevance2_{st.session_state.widget_suffix}"
)
# Reasoning Quality Slider
col2.markdown("**Reasoning Quality (1–5)**")
col2.markdown(reasoning_def)
col2.slider(
    "",
    1, 5, 3,
    key=f"reasoning2_{st.session_state.widget_suffix}"
)
# Practical Usefulness Slider
col2.markdown("**Practical Usefulness (1–5)**")
col2.markdown(usefulness_def)
col2.slider(
    "",
    1, 5, 3,
    key=f"usefulness2_{st.session_state.widget_suffix}"
)

# --- Navigation Buttons ---
nav_prev, nav_next = st.columns([1, 1])
nav_prev.button("⬅️ Previous", on_click=go_previous)
nav_next.button("Next ➡️", on_click=go_next)

