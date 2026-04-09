"""
Green Consumer Behavior Prediction

Streamlit-based ML application that predicts eco-friendly purchasing behavior
using classification models trained on synthetic consumer data.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="GreenSense · ML Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling (glassmorphism, typography, and layout overrides)
st.markdown("""
<style>
/* ── Google Font Import ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Root Variables ── */
:root {
  --green-50:  #f0fdf4;
  --green-100: #dcfce7;
  --green-400: #4ade80;
  --green-500: #22c55e;
  --green-600: #16a34a;
  --green-700: #15803d;
  --gray-50:   #f9fafb;
  --gray-100:  #f3f4f6;
  --gray-200:  #e5e7eb;
  --gray-400:  #9ca3af;
  --gray-600:  #4b5563;
  --gray-800:  #1f2937;
  --gray-900:  #111827;
  --radius-sm: 10px;
  --radius:    16px;
  --radius-lg: 24px;
  --shadow-sm: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
  --shadow:    0 4px 24px rgba(0,0,0,.07), 0 1px 4px rgba(0,0,0,.04);
  --shadow-lg: 0 12px 48px rgba(0,0,0,.10), 0 4px 12px rgba(0,0,0,.06);
  --glass-bg:  rgba(255,255,255,0.72);
  --glass-bdr: rgba(255,255,255,0.55);
}

/* ── Base Reset ── */
html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(145deg, #f0fdf4 0%, #f8fafc 40%, #f0f9f0 100%) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--gray-800) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.82) !important;
  backdrop-filter: blur(24px) saturate(160%) !important;
  -webkit-backdrop-filter: blur(24px) saturate(160%) !important;
  border-right: 1px solid var(--gray-200) !important;
  box-shadow: 2px 0 24px rgba(0,0,0,.05) !important;
}

[data-testid="stSidebar"] * { color: var(--gray-800) !important; }

[data-testid="stSidebar"] .stRadio label {
  padding: 8px 14px !important;
  border-radius: var(--radius-sm) !important;
  cursor: pointer !important;
  transition: background .18s ease !important;
  font-weight: 500 !important;
  font-size: 0.9rem !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
  background: var(--green-100) !important;
}

/* ── Main Container ── */
.main .block-container {
  max-width: 1180px !important;
  padding: 2rem 2.5rem 4rem !important;
  margin: 0 auto !important;
}

/* ── Hero Header ── */
.hero-wrap {
  text-align: center;
  padding: 3.5rem 2rem 2.5rem;
  margin-bottom: 2rem;
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--green-100);
  color: var(--green-700);
  border: 1px solid var(--green-400);
  border-radius: 99px;
  padding: 5px 14px;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 1.2rem;
}

.hero-title {
  font-family: 'DM Serif Display', serif;
  font-size: clamp(2.4rem, 5vw, 3.6rem);
  font-weight: 400;
  line-height: 1.15;
  color: var(--gray-900);
  margin: 0 0 0.6rem;
}

.hero-title span {
  background: linear-gradient(135deg, #16a34a 0%, #4ade80 60%, #22d3ee 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-subtitle {
  font-size: 1.05rem;
  color: var(--gray-400);
  font-weight: 400;
  max-width: 560px;
  margin: 0 auto;
  line-height: 1.65;
}

/* ── Card ── */
.card {
  background: var(--glass-bg);
  backdrop-filter: blur(20px) saturate(150%);
  -webkit-backdrop-filter: blur(20px) saturate(150%);
  border: 1px solid var(--glass-bdr);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
  padding: 2rem 2.2rem;
  margin-bottom: 1.6rem;
  transition: box-shadow .25s ease, transform .25s ease;
}

.card:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
}

.card-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--gray-800);
  margin: 0 0 0.3rem;
  display: flex;
  align-items: center;
  gap: 8px;
}

.card-subtitle {
  font-size: 0.83rem;
  color: var(--gray-400);
  margin: 0 0 1.4rem;
  font-weight: 400;
}

/* ── Section Divider ── */
.section-divider {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 2.4rem 0 1.6rem;
}

.section-divider h3 {
  font-family: 'DM Serif Display', serif;
  font-size: 1.4rem;
  font-weight: 400;
  color: var(--gray-800);
  margin: 0;
  white-space: nowrap;
}

.section-divider .line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--green-200), transparent);
}

/* ── Metric Pill ── */
.metric-pill {
  background: var(--glass-bg);
  border: 1px solid var(--glass-bdr);
  border-radius: var(--radius);
  padding: 1.2rem 1.4rem;
  text-align: center;
  box-shadow: var(--shadow-sm);
  transition: transform .2s ease, box-shadow .2s ease;
}

.metric-pill:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow);
}

.metric-pill .metric-val {
  font-family: 'DM Serif Display', serif;
  font-size: 2.1rem;
  color: var(--green-600);
  line-height: 1;
  margin-bottom: 4px;
}

.metric-pill .metric-label {
  font-size: 0.78rem;
  color: var(--gray-400);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* ── Model Comparison Table ── */
.model-row {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  transition: background .18s;
  margin-bottom: 4px;
}

.model-row:hover { background: var(--green-50); }

.model-row.best { background: var(--green-100); border: 1px solid var(--green-300); }

.model-badge {
  background: var(--green-500);
  color: white;
  font-size: 0.68rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 99px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

/* ── Prediction Result ── */
.result-card-positive {
  background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%);
  border: 1.5px solid var(--green-400);
  border-radius: var(--radius-lg);
  padding: 2.8rem 2.4rem;
  text-align: center;
  box-shadow: 0 8px 32px rgba(34,197,94,.15);
}

.result-card-negative {
  background: linear-gradient(135deg, #fef2f2 0%, #fff5f5 100%);
  border: 1.5px solid #fca5a5;
  border-radius: var(--radius-lg);
  padding: 2.8rem 2.4rem;
  text-align: center;
  box-shadow: 0 8px 32px rgba(239,68,68,.10);
}

.result-label {
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--gray-400);
  margin-bottom: 0.6rem;
}

.result-value {
  font-family: 'DM Serif Display', serif;
  font-size: 2.6rem;
  font-weight: 400;
  margin-bottom: 0.4rem;
}

.confidence-tag {
  display: inline-block;
  padding: 4px 14px;
  border-radius: 99px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-top: 6px;
}

.confidence-high   { background: #dcfce7; color: #15803d; }
.confidence-medium { background: #fef9c3; color: #854d0e; }
.confidence-low    { background: #fee2e2; color: #991b1b; }

/* ── Progress Bar Override ── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--green-500), var(--green-400)) !important;
  border-radius: 99px !important;
}

/* ── Streamlit Widget Cleanup ── */
.stSelectbox > div > div,
.stSlider > div > div > div,
.stTextInput > div > div {
  border-radius: var(--radius-sm) !important;
}

.stButton > button {
  background: linear-gradient(135deg, var(--green-600), var(--green-500)) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.55rem 1.6rem !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 2px 12px rgba(22,163,74,.3) !important;
  transition: all .22s ease !important;
}

.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(22,163,74,.4) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--glass-bg) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  border: 1px solid var(--glass-bdr) !important;
}

/* ── DataFrame ── */
.stDataFrame { border-radius: var(--radius-sm) !important; overflow: hidden !important; }

/* ── Footer ── */
.footer {
  text-align: center;
  margin-top: 4rem;
  padding: 1.4rem;
  font-size: 0.78rem;
  color: var(--gray-400);
  border-top: 1px solid var(--gray-200);
}
</style>
""", unsafe_allow_html=True)


# Data loading and preprocessing

@st.cache_data
def load_data():
    """Generate a synthetic consumer dataset with 1,200 records and derived target labels."""
    np.random.seed(42)
    n = 1200

    age = np.random.randint(18, 70, n)
    income = np.random.choice(["Low", "Medium", "High"], n, p=[0.3, 0.45, 0.25])
    education = np.random.choice(["High School", "Bachelor's", "Master's", "PhD"], n, p=[0.25, 0.40, 0.25, 0.10])
    env_awareness = np.random.randint(1, 11, n)
    social_influence = np.random.randint(1, 11, n)
    eco_label_trust = np.random.randint(1, 11, n)
    price_sensitivity = np.random.randint(1, 11, n)
    purchase_freq = np.random.choice(["Rarely", "Sometimes", "Often", "Always"], n, p=[0.2, 0.3, 0.3, 0.2])
    region = np.random.choice(["Urban", "Suburban", "Rural"], n, p=[0.5, 0.3, 0.2])

    income_score = {"Low": 0, "Medium": 1, "High": 2}
    edu_score    = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    freq_score   = {"Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3}
    region_score = {"Urban": 2, "Suburban": 1, "Rural": 0}

    score = (
        0.30 * env_awareness
        + 0.20 * social_influence
        + 0.15 * eco_label_trust
        - 0.10 * price_sensitivity
        + 0.12 * np.array([income_score[i]  for i in income])   * 2
        + 0.08 * np.array([edu_score[i]     for i in education]) * 1.5
        + 0.10 * np.array([freq_score[i]    for i in purchase_freq]) * 2
        + 0.05 * np.array([region_score[i]  for i in region])
        + np.random.normal(0, 0.8, n)
    )

    behavior = (score > np.median(score)).astype(int)

    return pd.DataFrame({
        "Age": age,
        "Income": income,
        "Education": education,
        "Environmental_Awareness": env_awareness,
        "Social_Influence": social_influence,
        "Eco_Label_Trust": eco_label_trust,
        "Price_Sensitivity": price_sensitivity,
        "Purchase_Frequency": purchase_freq,
        "Region": region,
        "Green_Purchase_Behavior": behavior,
    })


def encode_data(df):
    """Label-encode categorical features and split into X, y with encoder references."""
    df_enc = df.copy()
    encoders = {}
    cat_cols = ["Income", "Education", "Purchase_Frequency", "Region"]
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    feature_cols = [
        "Age", "Income", "Education",
        "Environmental_Awareness", "Social_Influence",
        "Eco_Label_Trust", "Price_Sensitivity",
        "Purchase_Frequency", "Region",
    ]
    X = df_enc[feature_cols]
    y = df_enc["Green_Purchase_Behavior"]
    return X, y, encoders, feature_cols


# Model training and evaluation

@st.cache_data
def train_models(X_train, X_test, y_train, y_test):
    """Train five classifiers and return a dict of fitted models, metrics, and predictions."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = {
        "Logistic Regression":      LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        "Decision Tree":            DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":            RandomForestClassifier(n_estimators=150, random_state=42),
        "Gradient Boosting":        GradientBoostingClassifier(n_estimators=150, random_state=42),
        "Support Vector Machine":   SVC(kernel="rbf", probability=True, random_state=42),
    }

    results = {}
    for name, mdl in models.items():
        uses_scale = name in ("Logistic Regression", "Support Vector Machine")
        Xtr = X_tr_s if uses_scale else X_train
        Xte = X_te_s if uses_scale else X_test
        mdl.fit(Xtr, y_train)
        preds = mdl.predict(Xte)
        acc   = accuracy_score(y_test, preds)
        results[name] = {
            "model": mdl,
            "acc":   acc,
            "preds": preds,
            "scaler": scaler if uses_scale else None,
            "report": classification_report(y_test, preds, output_dict=True),
            "cm": confusion_matrix(y_test, preds),
        }

    return results


def confidence_label(prob: float) -> tuple[str, str]:
    """Map a prediction probability to a human-readable confidence label and CSS class."""
    if prob >= 0.80:
        return "High Confidence", "confidence-high"
    elif prob >= 0.60:
        return "Medium Confidence", "confidence-medium"
    else:
        return "Low Confidence", "confidence-low"


# Chart styling utilities

GREEN_PALETTE = ["#22c55e", "#4ade80", "#86efac", "#bbf7d0", "#dcfce7"]
ACCENT = "#16a34a"


def style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply consistent styling to a matplotlib axes object."""
    ax.set_facecolor("#fafafa")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e5e7eb")
    ax.tick_params(colors="#6b7280", labelsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1f2937", pad=12)
    ax.set_xlabel(xlabel, fontsize=9, color="#9ca3af", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=9, color="#9ca3af", labelpad=8)


def fig_to_st(fig):
    """Render a matplotlib figure in Streamlit and clean up."""
    fig.patch.set_facecolor("none")
    st.pyplot(fig)
    plt.close(fig)


# Sidebar navigation

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 1rem;'>
      <div style='font-size:2rem;'>🌿</div>
      <div style='font-weight:700; font-size:1.05rem; color:#15803d; margin-top:4px;'>GreenSense</div>
      <div style='font-size:0.75rem; color:#9ca3af; margin-top:2px;'>ML Insights Platform</div>
    </div>
    <hr style='border:none;border-top:1px solid #e5e7eb;margin:0.6rem 0 1rem;'>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigate",
        ["🏠  Overview", "🔬  Explore Data", "🤖  Train Models", "🎯  Predict", "📊  Insights"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style='border:none;border-top:1px solid #e5e7eb;margin:1.4rem 0 1rem;'>
    <div style='font-size:0.75rem; color:#9ca3af; padding:0 0.4rem;'>
      <b style='color:#374151;'>Dataset</b><br>
      1,200 synthetic consumer records<br><br>
      <b style='color:#374151;'>Models</b><br>
      Logistic Regression · Decision Tree<br>
      Random Forest · Gradient Boosting<br>
      Support Vector Machine
    </div>
    """, unsafe_allow_html=True)


# Initialize data and train/test split

df = load_data()
X, y, encoders, feature_cols = encode_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# --- Page: Overview ---

if nav == "🏠  Overview":
    st.markdown("""
    <div class='hero-wrap'>
      <div class='hero-badge'>🌿 Sustainability Intelligence</div>
      <h1 class='hero-title'>
        Predict <span>Green Consumer</span><br>Behavior with ML
      </h1>
      <p class='hero-subtitle'>
        A modern machine learning platform for understanding eco-conscious purchasing patterns.
        Train models, explore data, and generate real-time predictions.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Summary metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-pill'>
          <div class='metric-val'>{len(df):,}</div>
          <div class='metric-label'>Total Records</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-pill'>
          <div class='metric-val'>{len(feature_cols)}</div>
          <div class='metric-label'>Features</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        pct = int(df["Green_Purchase_Behavior"].mean() * 100)
        st.markdown(f"""
        <div class='metric-pill'>
          <div class='metric-val'>{pct}%</div>
          <div class='metric-label'>Green Consumers</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-pill'>
          <div class='metric-val'>5</div>
          <div class='metric-label'>ML Models</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview charts
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class='card'>
          <div class='card-title'>📊 Target Distribution</div>
          <div class='card-subtitle'>Green vs. Non-Green consumer split</div>
        """, unsafe_allow_html=True)
        counts = df["Green_Purchase_Behavior"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 2.8))
        bars = ax.bar(
            ["Non-Green", "Green"],
            [counts[0], counts[1]],
            color=["#d1fae5", "#22c55e"],
            edgecolor="white",
            linewidth=2,
            width=0.5,
            zorder=3,
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#e5e7eb", zorder=0)
        style_ax(ax, ylabel="Count")
        for bar, val in zip(bars, [counts[0], counts[1]]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                    str(val), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#374151")
        fig.tight_layout()
        fig_to_st(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='card'>
          <div class='card-title'>🧬 Environmental Awareness</div>
          <div class='card-subtitle'>Distribution across all respondents</div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 2.8))
        ax.hist(df["Environmental_Awareness"], bins=10, color="#22c55e",
                edgecolor="white", linewidth=1.5, rwidth=0.85, zorder=3, alpha=0.85)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#e5e7eb", zorder=0)
        style_ax(ax, xlabel="Score (1–10)", ylabel="Frequency")
        fig.tight_layout()
        fig_to_st(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # How-it-works workflow steps
    st.markdown("""
    <div class='section-divider'>
      <h3>How it works</h3><div class='line'></div>
    </div>
    <div class='card'>
      <div style='display:flex; gap:1.8rem; flex-wrap:wrap;'>
        <div style='flex:1; min-width:160px; text-align:center; padding:1rem;'>
          <div style='font-size:2rem; margin-bottom:.5rem;'>🔬</div>
          <div style='font-weight:700; font-size:.95rem; color:#1f2937;'>Explore Data</div>
          <div style='font-size:.82rem; color:#9ca3af; margin-top:4px;'>Visualize patterns & distributions</div>
        </div>
        <div style='width:1px; background:#e5e7eb;'></div>
        <div style='flex:1; min-width:160px; text-align:center; padding:1rem;'>
          <div style='font-size:2rem; margin-bottom:.5rem;'>🤖</div>
          <div style='font-weight:700; font-size:.95rem; color:#1f2937;'>Train Models</div>
          <div style='font-size:.82rem; color:#9ca3af; margin-top:4px;'>Compare 5 ML algorithms</div>
        </div>
        <div style='width:1px; background:#e5e7eb;'></div>
        <div style='flex:1; min-width:160px; text-align:center; padding:1rem;'>
          <div style='font-size:2rem; margin-bottom:.5rem;'>🎯</div>
          <div style='font-weight:700; font-size:.95rem; color:#1f2937;'>Predict</div>
          <div style='font-size:.82rem; color:#9ca3af; margin-top:4px;'>Get real-time predictions</div>
        </div>
        <div style='width:1px; background:#e5e7eb;'></div>
        <div style='flex:1; min-width:160px; text-align:center; padding:1rem;'>
          <div style='font-size:2rem; margin-bottom:.5rem;'>📊</div>
          <div style='font-weight:700; font-size:.95rem; color:#1f2937;'>Insights</div>
          <div style='font-size:.82rem; color:#9ca3af; margin-top:4px;'>Feature importance & impact</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# --- Page: Explore Data ---

elif nav == "🔬  Explore Data":
    st.markdown("""
    <div class='section-divider'>
      <h3>Explore Dataset</h3><div class='line'></div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset preview table
    st.markdown("<div class='card'><div class='card-title'>📋 Raw Data</div><div class='card-subtitle'>First 50 records of the dataset</div>", unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

    # Descriptive statistics
    st.markdown("<div class='card'><div class='card-title'>📐 Summary Statistics</div><div class='card-subtitle'>Descriptive statistics for numerical features</div>", unsafe_allow_html=True)
    st.dataframe(df.describe().T.style.background_gradient(cmap="Greens"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Numerical feature distributions by target class
    st.markdown("""
    <div class='section-divider'>
      <h3>Feature Distributions</h3><div class='line'></div>
    </div>
    """, unsafe_allow_html=True)

    num_feats = ["Age", "Environmental_Awareness", "Social_Influence", "Eco_Label_Trust", "Price_Sensitivity"]
    fig, axes = plt.subplots(1, len(num_feats), figsize=(14, 3.2))
    for ax, feat in zip(axes, num_feats):
        for val, color in [(0, "#d1fae5"), (1, "#22c55e")]:
            ax.hist(df[df["Green_Purchase_Behavior"] == val][feat],
                    bins=12, alpha=0.7, color=color, edgecolor="white", linewidth=1)
        style_ax(ax, title=feat.replace("_", " "), xlabel="Value", ylabel="")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#e5e7eb")

    patch0 = mpatches.Patch(color="#d1fae5", label="Non-Green")
    patch1 = mpatches.Patch(color="#22c55e", label="Green")
    fig.legend(handles=[patch0, patch1], loc="upper right", fontsize=8,
               frameon=True, edgecolor="#e5e7eb")
    fig.tight_layout(pad=1.5)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    fig_to_st(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Green purchase rate by categorical features
    st.markdown("""
    <div class='section-divider'>
      <h3>Categorical Features</h3><div class='line'></div>
    </div>
    """, unsafe_allow_html=True)

    cat_feats = ["Income", "Education", "Purchase_Frequency", "Region"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, feat in zip(axes.flatten(), cat_feats):
        ct = df.groupby(feat)["Green_Purchase_Behavior"].mean().sort_values(ascending=False)
        bars = ax.bar(ct.index, ct.values * 100, color=GREEN_PALETTE[:len(ct)],
                      edgecolor="white", linewidth=1.5, zorder=3, width=0.55)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#e5e7eb", zorder=0)
        ax.axhline(50, color="#f87171", linewidth=1, linestyle="--", alpha=0.7)
        for bar, val in zip(bars, ct.values * 100):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#374151")
        style_ax(ax, title=f"% Green by {feat.replace('_', ' ')}", ylabel="%")
        ax.set_xticklabels(ct.index, rotation=15, ha="right", fontsize=8)
    fig.tight_layout(pad=2)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    fig_to_st(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Correlation matrix for numerical features
    st.markdown("""
    <div class='section-divider'>
      <h3>Correlation Heatmap</h3><div class='line'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    num_df = df[["Age", "Environmental_Awareness", "Social_Influence",
                 "Eco_Label_Trust", "Price_Sensitivity", "Green_Purchase_Behavior"]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt=".2f", cmap="Greens",
                linewidths=0.5, linecolor="#f3f4f6", ax=ax,
                annot_kws={"size": 9, "weight": "bold"},
                cbar_kws={"shrink": .7})
    ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold", color="#1f2937", pad=14)
    ax.tick_params(colors="#6b7280", labelsize=8.5)
    fig.tight_layout()
    fig_to_st(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# --- Page: Train Models ---

elif nav == "🤖  Train Models":
    st.markdown("""
    <div class='section-divider'>
      <h3>Model Training &amp; Comparison</h3><div class='line'></div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("⚙️  Training models — this takes a moment…"):
        trained = train_models(X_train, X_test, y_train, y_test)

    best_name = max(trained, key=lambda m: trained[m]["acc"])
    best_acc  = trained[best_name]["acc"]

    # Highlight the top-performing model
    st.markdown(f"""
    <div class='card' style='background:linear-gradient(135deg,#dcfce7,#f0fdf4);
         border:1.5px solid #86efac; text-align:center;'>
      <div style='font-size:.75rem;font-weight:700;text-transform:uppercase;
           letter-spacing:.1em;color:#9ca3af;margin-bottom:.4rem;'>🏆 Best Performing Model</div>
      <div style='font-family:"DM Serif Display",serif;font-size:2rem;color:#15803d;'>
        {best_name}
      </div>
      <div style='font-size:1.5rem;font-weight:700;color:#16a34a;margin-top:.2rem;'>
        {best_acc:.1%} Accuracy
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Ranked accuracy comparison with progress bars
    st.markdown("<div class='card'><div class='card-title'>📊 Accuracy Comparison</div><div class='card-subtitle'>Test-set accuracy across all models</div>", unsafe_allow_html=True)
    sorted_models = sorted(trained.items(), key=lambda x: x[1]["acc"], reverse=True)
    for name, info in sorted_models:
        acc = info["acc"]
        is_best = name == best_name
        badge = "<span class='model-badge'>BEST</span>" if is_best else ""
        row_cls = "model-row best" if is_best else "model-row"
        bar_w = acc * 100
        st.markdown(f"""
        <div class='{row_cls}'>
          <div style='width:180px; font-weight:{"700" if is_best else "500"};
               font-size:.88rem; color:#1f2937;'>{name}</div>
          {badge}
          <div style='flex:1; background:#e5e7eb; border-radius:99px; height:8px; overflow:hidden;'>
            <div style='width:{bar_w:.1f}%; height:100%;
                 background:linear-gradient(90deg,#16a34a,#4ade80);
                 border-radius:99px;'></div>
          </div>
          <div style='font-weight:700; font-size:.92rem; color:#15803d;
               min-width:56px; text-align:right;'>{acc:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Visual accuracy bar chart and classification report
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("<div class='card'><div class='card-title'>📈 Visual Comparison</div><div class='card-subtitle'>Accuracy bars</div>", unsafe_allow_html=True)
        names = [n.replace(" ", "\n") for n, _ in sorted_models]
        accs  = [info["acc"] for _, info in sorted_models]
        colors = [ACCENT if n.replace("\n", " ") == best_name else "#86efac" for n in names]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=1.5, zorder=3, width=0.55)
        ax.set_ylim(0.5, 1.0)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#e5e7eb", zorder=0)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                    f"{val:.2%}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#374151")
        style_ax(ax, ylabel="Accuracy")
        fig.tight_layout()
        fig_to_st(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='card'><div class='card-title'>📋 Best Model Report</div><div class='card-subtitle'>{best_name}</div>", unsafe_allow_html=True)
        report = trained[best_name]["report"]
        for label, vals in report.items():
            if isinstance(vals, dict):
                name_display = {"0": "Non-Green", "1": "Green"}.get(label, label.title())
                p = vals.get("precision", 0)
                r = vals.get("recall", 0)
                f = vals.get("f1-score", 0)
                st.markdown(f"""
                <div style='padding:8px 0; border-bottom:1px solid #f3f4f6;'>
                  <div style='font-weight:700;font-size:.85rem;color:#374151;'>{name_display}</div>
                  <div style='display:flex;gap:14px;font-size:.8rem;color:#6b7280;margin-top:3px;'>
                    <span>Precision: <b style='color:#16a34a;'>{p:.2f}</b></span>
                    <span>Recall: <b style='color:#16a34a;'>{r:.2f}</b></span>
                    <span>F1: <b style='color:#16a34a;'>{f:.2f}</b></span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Confusion matrix heatmap for the best model
    st.markdown("<div class='card'><div class='card-title'>🔲 Confusion Matrix</div><div class='card-subtitle'>" + best_name + "</div>", unsafe_allow_html=True)
    cm = trained[best_name]["cm"]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Non-Green", "Green"],
                yticklabels=["Non-Green", "Green"],
                linewidths=1, linecolor="#f3f4f6",
                ax=ax, annot_kws={"size": 13, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=9, color="#9ca3af")
    ax.set_ylabel("Actual",    fontsize=9, color="#9ca3af")
    ax.tick_params(colors="#6b7280", labelsize=8.5)
    ax.set_title("", pad=0)
    fig.tight_layout()
    fig_to_st(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# --- Page: Predict ---

elif nav == "🎯  Predict":
    st.markdown("""
    <div class='section-divider'>
      <h3>Live Prediction</h3><div class='line'></div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Loading models…"):
        trained = train_models(X_train, X_test, y_train, y_test)

    best_name = max(trained, key=lambda m: trained[m]["acc"])

    st.markdown(f"""
    <div class='card' style='margin-bottom:1.2rem;'>
      <div class='card-title'>⚡ Active Model</div>
      <div style='display:flex;align-items:center;gap:10px;margin-top:6px;'>
        <span style='font-size:1.1rem;font-weight:700;color:#15803d;'>{best_name}</span>
        <span class='model-badge'>AUTO-SELECTED BEST</span>
      </div>
      <div style='font-size:.82rem;color:#9ca3af;margin-top:4px;'>
        Accuracy: {trained[best_name]['acc']:.2%}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Consumer profile input form
    st.markdown("<div class='card'><div class='card-title'>🧾 Consumer Profile</div><div class='card-subtitle'>Enter features to generate a prediction</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age       = st.slider("🎂 Age", 18, 70, 30)
        income    = st.selectbox("💰 Income Level", ["Low", "Medium", "High"])
        education = st.selectbox("🎓 Education", ["High School", "Bachelor's", "Master's", "PhD"])
    with c2:
        env_aw    = st.slider("🌱 Environmental Awareness", 1, 10, 7)
        soc_inf   = st.slider("👥 Social Influence", 1, 10, 6)
        eco_trust = st.slider("🏷️ Eco-Label Trust", 1, 10, 7)
    with c3:
        price_s   = st.slider("💲 Price Sensitivity", 1, 10, 5)
        purch_freq= st.selectbox("🛒 Purchase Frequency", ["Rarely", "Sometimes", "Often", "Always"])
        region    = st.selectbox("📍 Region", ["Urban", "Suburban", "Rural"])

    predict_btn = st.button("🎯  Generate Prediction", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        input_dict = {
            "Age": age,
            "Income": encoders["Income"].transform([income])[0],
            "Education": encoders["Education"].transform([education])[0],
            "Environmental_Awareness": env_aw,
            "Social_Influence": soc_inf,
            "Eco_Label_Trust": eco_trust,
            "Price_Sensitivity": price_s,
            "Purchase_Frequency": encoders["Purchase_Frequency"].transform([purch_freq])[0],
            "Region": encoders["Region"].transform([region])[0],
        }
        input_df = pd.DataFrame([input_dict])
        model_info = trained[best_name]
        mdl = model_info["model"]
        scaler = model_info["scaler"]
        X_in = scaler.transform(input_df) if scaler else input_df
        pred = mdl.predict(X_in)[0]
        prob = mdl.predict_proba(X_in)[0][pred]
        conf_label, conf_cls = confidence_label(prob)

        if pred == 1:
            st.markdown(f"""
            <div class='result-card-positive'>
              <div class='result-label'>Prediction Result</div>
              <div class='result-value' style='color:#15803d;'>🌿 Green Consumer</div>
              <div style='color:#374151;font-size:.95rem;margin:.4rem 0;'>
                This consumer is likely to make eco-friendly purchases.
              </div>
              <span class='confidence-tag {conf_cls}'>{conf_label}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card-negative'>
              <div class='result-label'>Prediction Result</div>
              <div class='result-value' style='color:#dc2626;'>❌ Non-Green Consumer</div>
              <div style='color:#374151;font-size:.95rem;margin:.4rem 0;'>
                This consumer is unlikely to prioritize eco-friendly purchases.
              </div>
              <span class='confidence-tag {conf_cls}'>{conf_label}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><div class='card-title'>📊 Prediction Confidence: {prob:.1%}</div>", unsafe_allow_html=True)
        st.progress(float(prob))
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;font-size:.8rem;color:#9ca3af;margin-top:4px;'>
          <span>0%</span><span>50%</span><span>100%</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# --- Page: Insights ---

elif nav == "📊  Insights":
    st.markdown("""
    <div class='section-divider'>
      <h3>Feature Importance &amp; Insights</h3><div class='line'></div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Loading models…"):
        trained = train_models(X_train, X_test, y_train, y_test)

    # Extract feature importance scores from the Random Forest model
    rf_model = trained["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

    st.markdown("<div class='card'><div class='card-title'>🌲 Feature Importance</div><div class='card-subtitle'>From Random Forest — which factors drive green behavior most?</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors_bar = [ACCENT if v == feat_imp.max() else "#86efac" for v in feat_imp.values]
    bars = ax.barh(
        [f.replace("_", " ") for f in feat_imp.index],
        feat_imp.values,
        color=colors_bar,
        edgecolor="white",
        linewidth=1.5,
        height=0.65,
        zorder=3,
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="#e5e7eb", zorder=0)
    for bar, val in zip(bars, feat_imp.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, fontweight="bold", color="#374151")
    style_ax(ax, xlabel="Importance Score")
    fig.tight_layout()
    fig_to_st(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Display top three most influential features
    top3 = feat_imp.sort_values(ascending=False).head(3)
    st.markdown("""
    <div class='section-divider'>
      <h3>Top Drivers</h3><div class='line'></div>
    </div>
    """, unsafe_allow_html=True)

    icons = ["🥇", "🥈", "🥉"]
    cols  = st.columns(3)
    for col, (feat, val), icon in zip(cols, top3.items(), icons):
        with col:
            st.markdown(f"""
            <div class='metric-pill'>
              <div style='font-size:1.6rem;'>{icon}</div>
              <div style='font-weight:700;font-size:.95rem;color:#1f2937;margin:.5rem 0 .2rem;'>
                {feat.replace('_', ' ')}
              </div>
              <div class='metric-val' style='font-size:1.5rem;'>{val:.3f}</div>
              <div class='metric-label'>importance</div>
            </div>
            """, unsafe_allow_html=True)

    # Behavioral pattern visualizations
    st.markdown("""
    <div class='section-divider'>
      <h3>Behavioral Patterns</h3><div class='line'></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'><div class='card-title'>🌱 Awareness vs Social Influence</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        colors = df["Green_Purchase_Behavior"].map({0: "#d1fae5", 1: "#16a34a"})
        ax.scatter(df["Environmental_Awareness"], df["Social_Influence"],
                   c=colors, alpha=0.55, s=18, edgecolors="white", linewidths=0.5, zorder=3)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, color="#e5e7eb", zorder=0)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#e5e7eb", zorder=0)
        style_ax(ax, xlabel="Environmental Awareness", ylabel="Social Influence")
        p0 = mpatches.Patch(color="#d1fae5", label="Non-Green")
        p1 = mpatches.Patch(color="#16a34a", label="Green")
        ax.legend(handles=[p0, p1], fontsize=8, frameon=True, edgecolor="#e5e7eb")
        fig.tight_layout()
        fig_to_st(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'><div class='card-title'>💲 Price Sensitivity by Behavior</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        g0 = df[df["Green_Purchase_Behavior"] == 0]["Price_Sensitivity"]
        g1 = df[df["Green_Purchase_Behavior"] == 1]["Price_Sensitivity"]
        bp = ax.boxplot([g0, g1], labels=["Non-Green", "Green"],
                        patch_artist=True, notch=True, widths=0.45,
                        medianprops=dict(color="white", linewidth=2))
        colors_box = ["#d1fae5", "#22c55e"]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#e5e7eb", zorder=0)
        style_ax(ax, ylabel="Price Sensitivity Score")
        fig.tight_layout()
        fig_to_st(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
  🌿 GreenSense · Machine Learning Platform &nbsp;·&nbsp;
  Built with Streamlit &amp; scikit-learn &nbsp;·&nbsp;
  Data is synthetic &amp; for demonstration purposes
</div>
""", unsafe_allow_html=True)
