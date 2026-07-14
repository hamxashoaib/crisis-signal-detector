import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file
import numpy as np
import plotly.graph_objects as go
import time
import os

# ─────────────────────────────────────────────
#  CONFIG — edit these to match your model
# ─────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"   # base architecture you fine-tuned
WEIGHTS_PATH = "model.safetensors" # path to your downloaded weights file
NUM_LABELS = 2                     # change to your number of classes
LABEL_NAMES = ["Negative", "Positive"]  # replace with your actual class names
MAX_LENGTH = 128
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="BERT Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}
.main { background: #0d0f14; }

.stApp {
    background: linear-gradient(135deg, #0d0f14 0%, #111827 100%);
    color: #e2e8f0;
}

.metric-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #6366f1; }
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #818cf8;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

.result-box {
    background: #1a1f2e;
    border: 1px solid #6366f1;
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
}
.result-label {
    font-size: 2rem;
    font-weight: 700;
    color: #a5b4fc;
    font-family: 'JetBrains Mono', monospace;
}
.confidence-badge {
    display: inline-block;
    background: #312e81;
    color: #c7d2fe;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 8px;
}

.stTextArea textarea {
    background: #1a1f2e !important;
    color: #e2e8f0 !important;
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.3) !important;
}

div[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1f2937;
}

.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 12px;
    margin-top: 24px;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load tokenizer + BERT model with your fine-tuned weights."""
    if not os.path.exists(WEIGHTS_PATH):
        return None, None, f"❌ Weights file not found: `{WEIGHTS_PATH}`\n\nMake sure `model.safetensors` is in the same folder as `app.py`."

    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
        )
        state_dict = load_file(WEIGHTS_PATH)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, tokenizer, None
    except Exception as e:
        return None, None, f"❌ Error loading model: {e}"


@torch.no_grad()
def predict(text, model, tokenizer):
    """Run inference on a single text input."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 BERT Classifier")
    st.markdown("---")

    st.markdown('<div class="section-title">Model Info</div>', unsafe_allow_html=True)
    st.markdown(f"**Base:** `{MODEL_NAME}`")
    st.markdown(f"**Classes:** {NUM_LABELS}")
    st.markdown(f"**Max tokens:** {MAX_LENGTH}")

    st.markdown('<div class="section-title">Weights</div>', unsafe_allow_html=True)
    weights_exists = os.path.exists(WEIGHTS_PATH)
    if weights_exists:
        size_mb = os.path.getsize(WEIGHTS_PATH) / 1e6
        st.success(f"✅ `{WEIGHTS_PATH}` ({size_mb:.1f} MB)")
    else:
        st.error(f"Not found: `{WEIGHTS_PATH}`")

    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
    temperature = st.slider("Softmax temperature", 0.5, 2.0, 1.0, 0.1,
                            help="Higher = softer probabilities")

    st.markdown("---")
    st.markdown(
        "<small style='color:#4b5563'>Place `model.safetensors` in the same directory as `app.py`, then run:<br>"
        "`streamlit run app.py`</small>",
        unsafe_allow_html=True,
    )


# ── Load model ───────────────────────────────
with st.spinner("Loading model weights…"):
    model, tokenizer, load_error = load_model()


# ── Main UI ──────────────────────────────────
st.markdown("# Text Classification Dashboard")
st.markdown(
    "<span style='color:#64748b;font-size:0.95rem'>Fine-tuned BERT &nbsp;·&nbsp; Hugging Face Transformers &nbsp;·&nbsp; SafeTensors</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

if load_error:
    st.error(load_error)
    st.markdown("### Quick Fix")
    st.code(f"""
# Make sure model.safetensors is in this folder:
ls -lh {WEIGHTS_PATH}

# Install dependencies:
pip install streamlit transformers safetensors torch plotly
    """, language="bash")
    st.stop()

# Stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        '<div class="metric-card"><div class="value">BERT</div><div class="label">Architecture</div></div>',
        unsafe_allow_html=True)
with col2:
    st.markdown(
        f'<div class="metric-card"><div class="value">{NUM_LABELS}</div><div class="label">Classes</div></div>',
        unsafe_allow_html=True)
with col3:
    st.markdown(
        f'<div class="metric-card"><div class="value">{MAX_LENGTH}</div><div class="label">Max Tokens</div></div>',
        unsafe_allow_html=True)
with col4:
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    st.markdown(
        f'<div class="metric-card"><div class="value">{param_count:.0f}M</div><div class="label">Parameters</div></div>',
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📋 Batch Prediction"])

# ── TAB 1: Single ────────────────────────────
with tab1:
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Input Text</div>', unsafe_allow_html=True)
        user_text = st.text_area(
            label="",
            placeholder="Type or paste your text here…",
            height=180,
            label_visibility="collapsed",
        )

        run_btn = st.button("Run Prediction →", type="primary", use_container_width=True)

        # Example texts
        st.markdown('<div class="section-title">Try an example</div>', unsafe_allow_html=True)
        examples = [
            "This product is absolutely amazing! I love it.",
            "Terrible experience, would not recommend.",
            "It was okay, nothing special.",
        ]
        for ex in examples:
            if st.button(f"💬 {ex[:55]}…" if len(ex) > 55 else f"💬 {ex}", key=ex):
                user_text = ex
                run_btn = True

    with col_right:
        if run_btn and user_text.strip():
            with st.spinner("Running inference…"):
                t0 = time.time()
                pred_idx, probs = predict(user_text.strip(), model, tokenizer)
                # apply temperature
                logits_t = np.log(probs + 1e-9) / temperature
                probs_t = np.exp(logits_t) / np.exp(logits_t).sum()
                elapsed = (time.time() - t0) * 1000

            pred_label = LABEL_NAMES[pred_idx] if pred_idx < len(LABEL_NAMES) else f"Class {pred_idx}"
            confidence = float(probs_t[pred_idx]) * 100

            st.markdown(
                f"""<div class="result-box">
                    <div style="font-size:0.75rem;color:#6366f1;text-transform:uppercase;letter-spacing:.15em;margin-bottom:8px">Prediction</div>
                    <div class="result-label">{pred_label}</div>
                    <div class="confidence-badge">{confidence:.1f}% confidence</div>
                    <div style="font-size:0.75rem;color:#4b5563;margin-top:14px">⏱ {elapsed:.1f} ms</div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Probability bar chart
            st.markdown("<br>", unsafe_allow_html=True)
            labels = [LABEL_NAMES[i] if i < len(LABEL_NAMES) else f"Class {i}" for i in range(NUM_LABELS)]
            colors = ["#6366f1" if i == pred_idx else "#2d3748" for i in range(NUM_LABELS)]

            fig = go.Figure(go.Bar(
                x=labels,
                y=[float(p) * 100 for p in probs_t],
                marker_color=colors,
                text=[f"{float(p)*100:.1f}%" for p in probs_t],
                textposition="outside",
                textfont=dict(color="#a5b4fc", family="JetBrains Mono"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Sora"),
                yaxis=dict(
                    title="Probability (%)",
                    gridcolor="#1f2937",
                    range=[0, 110],
                    tickcolor="#374151",
                ),
                xaxis=dict(tickcolor="#374151"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=220,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif run_btn:
            st.warning("Please enter some text first.")
        else:
            st.markdown(
                "<div style='text-align:center;color:#374151;padding:60px 0;font-size:0.9rem'>"
                "Enter text and click <b>Run Prediction</b>"
                "</div>",
                unsafe_allow_html=True,
            )


# ── TAB 2: Batch ─────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Batch Input (one sentence per line)</div>', unsafe_allow_html=True)
    batch_text = st.text_area(
        label="",
        placeholder="I love this movie\nThis was a waste of time\nPretty decent overall",
        height=200,
        label_visibility="collapsed",
        key="batch_input",
    )

    if st.button("Run Batch →", type="primary"):
        lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Add at least one line.")
        else:
            results = []
            progress = st.progress(0, text="Running…")
            for i, line in enumerate(lines):
                pred_idx, probs = predict(line, model, tokenizer)
                logits_t = np.log(probs + 1e-9) / temperature
                probs_t = np.exp(logits_t) / np.exp(logits_t).sum()
                label = LABEL_NAMES[pred_idx] if pred_idx < len(LABEL_NAMES) else f"Class {pred_idx}"
                results.append({
                    "Text": line,
                    "Prediction": label,
                    "Confidence": f"{float(probs_t[pred_idx])*100:.1f}%",
                })
                progress.progress((i + 1) / len(lines), text=f"Processing {i+1}/{len(lines)}")

            progress.empty()
            st.success(f"✅ Processed {len(results)} samples")
            st.dataframe(results, use_container_width=True)

            # Distribution chart
            from collections import Counter
            counts = Counter(r["Prediction"] for r in results)
            fig2 = go.Figure(go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                marker=dict(colors=["#6366f1", "#06b6d4", "#f59e0b", "#ef4444", "#10b981"]),
                hole=0.5,
                textfont=dict(family="JetBrains Mono", color="white"),
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                title=dict(text="Class Distribution", font=dict(color="#a5b4fc")),
                height=280,
                margin=dict(t=40, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)
