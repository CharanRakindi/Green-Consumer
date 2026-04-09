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

# Premium design system — glassmorphism, depth, and micro-interactions
st.markdown("""
<style>
/* ── Typography ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@400;500;600;700&display=swap');

/* ── Design Tokens ── */
:root {
  --green-50:  #f0fdf4;
  --green-100: #dcfce7;
  --green-200: #bbf7d0;
  --green-300: #86efac;
  --green-400: #4ade80;
  --green-500: #22c55e;
  --green-600: #16a34a;
  --green-700: #15803d;
  --green-800: #166534;
  --gray-50:   #f9fafb;
  --gray-100:  #f3f4f6;
  --gray-200:  #e5e7eb;
  --gray-300:  #d1d5db;
  --gray-400:  #9ca3af;
  --gray-500:  #6b7280;
  --gray-600:  #4b5563;
  --gray-700:  #374151;
  --gray-800:  #1f2937;
  --gray-900:  #111827;
  --radius-xs: 8px;
  --radius-sm: 12px;
  --radius:    16px;
  --radius-lg: 20px;
  --radius-xl: 28px;
  --shadow-xs: 0 1px 2px rgba(0,0,0,.04);
  --shadow-sm: 0 2px 8px rgba(0,0,0,.04), 0 1px 2px rgba(0,0,0,.03);
  --shadow:    0 4px 16px rgba(0,0,0,.06), 0 2px 4px rgba(0,0,0,.03);
  --shadow-md: 0 8px 32px rgba(0,0,0,.08), 0 2px 8px rgba(0,0,0,.04);
  --shadow-lg: 0 16px 48px rgba(0,0,0,.10), 0 4px 16px rgba(0,0,0,.05);
  --shadow-xl: 0 24px 64px rgba(0,0,0,.12), 0 8px 24px rgba(0,0,0,.06);
  --shadow-green: 0 8px 32px rgba(22,163,74,.12), 0 2px 8px rgba(22,163,74,.06);
  --glass-bg:  rgba(255,255,255,0.65);
  --glass-bg-strong: rgba(255,255,255,0.82);
  --glass-border: rgba(255,255,255,0.50);
  --glass-border-subtle: rgba(0,0,0,0.04);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
  --ease-out: cubic-bezier(0.22, 0.61, 0.36, 1);
  --dur: 0.3s;
}

/* ── Global Reset ── */
html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 80% 60% at 10% 0%, rgba(220,252,231,0.5) 0%, transparent 50%),
    radial-gradient(ellipse 60% 50% at 90% 100%, rgba(187,247,208,0.3) 0%, transparent 50%),
    radial-gradient(ellipse 50% 40% at 50% 50%, rgba(240,253,244,0.4) 0%, transparent 60%),
    linear-gradient(160deg, #f8fdf9 0%, #f3f8f5 30%, #f7f9fb 60%, #f5faf7 100%) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  color: var(--gray-800) !important;
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
}

[data-testid="stHeader"] {
  background: rgba(248,253,249,0.7) !important;
  backdrop-filter: blur(20px) saturate(180%) !important;
  -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
  border-bottom: 1px solid rgba(0,0,0,0.03) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,
    rgba(255,255,255,0.90) 0%,
    rgba(248,253,249,0.88) 50%,
    rgba(240,253,244,0.85) 100%) !important;
  backdrop-filter: blur(32px) saturate(180%) !important;
  -webkit-backdrop-filter: blur(32px) saturate(180%) !important;
  border-right: 1px solid rgba(0,0,0,0.05) !important;
  box-shadow: 4px 0 32px rgba(0,0,0,.03) !important;
}

[data-testid="stSidebar"] * {
  color: var(--gray-700) !important;
  font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] .stRadio > div {
  gap: 2px !important;
}

[data-testid="stSidebar"] .stRadio label {
  padding: 10px 16px !important;
  border-radius: var(--radius-sm) !important;
  cursor: pointer !important;
  transition: all var(--dur) var(--ease-out) !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  letter-spacing: -0.01em !important;
  border: 1px solid transparent !important;
  margin: 0 !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(22,163,74,0.06) !important;
  border-color: rgba(22,163,74,0.10) !important;
  transform: translateX(2px) !important;
}

[data-testid="stSidebar"] .stRadio label[data-checked="true"],
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
  background: rgba(22,163,74,0.08) !important;
  border-color: rgba(22,163,74,0.15) !important;
  color: var(--green-700) !important;
  font-weight: 600 !important;
}

/* ── Main Container ── */
.main .block-container {
  max-width: 1200px !important;
  padding: 2.5rem 3rem 5rem !important;
  margin: 0 auto !important;
}

/* ── Hero Section ── */
.hero-wrap {
  text-align: center;
  padding: 4.5rem 2rem 3rem;
  margin-bottom: 2.5rem;
  position: relative;
}

.hero-wrap::before {
  content: '';
  position: absolute;
  top: -60px;
  left: 50%;
  transform: translateX(-50%);
  width: 500px;
  height: 500px;
  background: radial-gradient(circle, rgba(74,222,128,0.12) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(220,252,231,0.7);
  color: var(--green-700);
  border: 1px solid rgba(74,222,128,0.35);
  border-radius: 99px;
  padding: 7px 18px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 1.5rem;
  position: relative;
  z-index: 1;
  backdrop-filter: blur(8px);
  transition: all var(--dur) var(--ease-out);
}

.hero-badge:hover {
  background: rgba(220,252,231,0.9);
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(22,163,74,0.12);
}

.hero-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: clamp(2.6rem, 5.5vw, 4rem);
  font-weight: 700;
  line-height: 1.1;
  color: var(--gray-900);
  margin: 0 0 1rem;
  letter-spacing: -0.03em;
  position: relative;
  z-index: 1;
}

.hero-title span {
  background: linear-gradient(135deg, #15803d 0%, #22c55e 40%, #4ade80 70%, #06b6d4 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-subtitle {
  font-size: 1.1rem;
  color: var(--gray-500);
  font-weight: 400;
  max-width: 580px;
  margin: 0 auto;
  line-height: 1.7;
  letter-spacing: -0.01em;
  position: relative;
  z-index: 1;
}

/* ── Glass Card ── */
.card {
  background: var(--glass-bg);
  backdrop-filter: blur(24px) saturate(160%);
  -webkit-backdrop-filter: blur(24px) saturate(160%);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow);
  padding: 2.2rem 2.4rem;
  margin-bottom: 1.8rem;
  transition: all var(--dur) var(--ease-out);
  position: relative;
  overflow: hidden;
}

.card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.8) 50%, transparent 100%);
  pointer-events: none;
}

.card:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-3px);
  border-color: rgba(74,222,128,0.2);
}

.card-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--gray-800);
  margin: 0 0 0.35rem;
  display: flex;
  align-items: center;
  gap: 10px;
  letter-spacing: -0.02em;
}

.card-subtitle {
  font-size: 0.82rem;
  color: var(--gray-400);
  margin: 0 0 1.6rem;
  font-weight: 400;
  letter-spacing: -0.01em;
}

/* ── Section Divider ── */
.section-divider {
  display: flex;
  align-items: center;
  gap: 16px;
  margin: 3rem 0 2rem;
}

.section-divider h3 {
  font-family: 'Playfair Display', serif;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--gray-800);
  margin: 0;
  white-space: nowrap;
  letter-spacing: -0.02em;
}

.section-divider .line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, rgba(22,163,74,0.15), transparent 80%);
}

/* ── Metric Pill ── */
.metric-pill {
  background: var(--glass-bg-strong);
  border: 1px solid rgba(0,0,0,0.04);
  border-radius: var(--radius-lg);
  padding: 1.5rem 1.6rem;
  text-align: center;
  box-shadow: var(--shadow-sm);
  transition: all var(--dur) var(--ease-spring);
  position: relative;
  overflow: hidden;
}

.metric-pill::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 20%;
  right: 20%;
  height: 3px;
  background: linear-gradient(90deg, var(--green-400), var(--green-500));
  border-radius: 99px;
  opacity: 0;
  transition: all var(--dur) var(--ease-out);
}

.metric-pill:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-md);
  border-color: rgba(22,163,74,0.12);
}

.metric-pill:hover::after {
  opacity: 1;
}

.metric-pill .metric-val {
  font-family: 'Playfair Display', serif;
  font-size: 2.4rem;
  color: var(--green-700);
  line-height: 1;
  margin-bottom: 6px;
  font-weight: 700;
  letter-spacing: -0.02em;
}

.metric-pill .metric-label {
  font-size: 0.72rem;
  color: var(--gray-400);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

/* ── Model Comparison Rows ── */
.model-row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 16px;
  border-radius: var(--radius-sm);
  transition: all 0.2s var(--ease-out);
  margin-bottom: 6px;
  border: 1px solid transparent;
}

.model-row:hover {
  background: rgba(240,253,244,0.6);
  border-color: rgba(22,163,74,0.06);
  transform: translateX(4px);
}

.model-row.best {
  background: linear-gradient(135deg, rgba(220,252,231,0.5), rgba(240,253,244,0.5));
  border: 1px solid rgba(134,239,172,0.4);
  box-shadow: 0 2px 12px rgba(22,163,74,0.06);
}

.model-badge {
  background: linear-gradient(135deg, var(--green-500), var(--green-600));
  color: white;
  font-size: 0.62rem;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 99px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  box-shadow: 0 2px 8px rgba(22,163,74,0.25);
}

/* ── Prediction Result Cards ── */
.result-card-positive {
  background: linear-gradient(145deg, rgba(220,252,231,0.8) 0%, rgba(240,253,244,0.9) 50%, rgba(255,255,255,0.7) 100%);
  border: 1.5px solid rgba(74,222,128,0.35);
  border-radius: var(--radius-xl);
  padding: 3.5rem 3rem;
  text-align: center;
  box-shadow: var(--shadow-md), 0 0 0 1px rgba(74,222,128,0.1);
  position: relative;
  overflow: hidden;
  animation: resultFadeIn 0.6s var(--ease-spring);
}

.result-card-positive::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at 50% 50%, rgba(74,222,128,0.08) 0%, transparent 50%);
  animation: resultGlow 3s ease-in-out infinite;
  pointer-events: none;
}

.result-card-negative {
  background: linear-gradient(145deg, rgba(254,242,242,0.8) 0%, rgba(255,245,245,0.9) 50%, rgba(255,255,255,0.7) 100%);
  border: 1.5px solid rgba(252,165,165,0.35);
  border-radius: var(--radius-xl);
  padding: 3.5rem 3rem;
  text-align: center;
  box-shadow: var(--shadow-md), 0 0 0 1px rgba(252,165,165,0.1);
  position: relative;
  overflow: hidden;
  animation: resultFadeIn 0.6s var(--ease-spring);
}

.result-card-negative::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at 50% 50%, rgba(252,165,165,0.06) 0%, transparent 50%);
  animation: resultGlow 3s ease-in-out infinite;
  pointer-events: none;
}

@keyframes resultFadeIn {
  from { opacity: 0; transform: translateY(16px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes resultGlow {
  0%, 100% { transform: translate(0%, 0%); }
  50%      { transform: translate(2%, -2%); }
}

.result-label {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--gray-400);
  margin-bottom: 0.8rem;
}

.result-value {
  font-family: 'Playfair Display', serif;
  font-size: 2.8rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  letter-spacing: -0.02em;
  line-height: 1.2;
}

.result-desc {
  color: var(--gray-600);
  font-size: 0.95rem;
  margin: 0.5rem 0 1rem;
  line-height: 1.6;
  font-weight: 400;
}

.confidence-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 18px;
  border-radius: 99px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-top: 8px;
  transition: all 0.2s ease;
}

.confidence-tag:hover { transform: scale(1.03); }

.confidence-high   { background: rgba(220,252,231,0.8); color: #15803d; border: 1px solid rgba(134,239,172,0.3); }
.confidence-medium { background: rgba(254,249,195,0.8); color: #854d0e; border: 1px solid rgba(250,204,21,0.3); }
.confidence-low    { background: rgba(254,226,226,0.8); color: #991b1b; border: 1px solid rgba(252,165,165,0.3); }

/* ── Progress Bar ── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--green-500), var(--green-400), var(--green-300)) !important;
  border-radius: 99px !important;
  box-shadow: 0 2px 8px rgba(22,163,74,0.2) !important;
}

.stProgress > div > div > div {
  background: rgba(0,0,0,0.04) !important;
  border-radius: 99px !important;
}

/* ── Streamlit Widget Overrides ── */
.stSelectbox > div > div {
  border-radius: var(--radius-sm) !important;
  border-color: rgba(0,0,0,0.08) !important;
  background: rgba(255,255,255,0.7) !important;
  transition: all 0.2s ease !important;
  font-size: 0.9rem !important;
}

.stSelectbox > div > div:hover,
.stSelectbox > div > div:focus-within {
  border-color: rgba(22,163,74,0.25) !important;
  box-shadow: 0 0 0 3px rgba(22,163,74,0.06) !important;
  background: rgba(255,255,255,0.9) !important;
}

.stSlider > div > div > div {
  border-radius: 99px !important;
}

.stSlider > div > div > div > div[role="slider"] {
  background: var(--green-500) !important;
  border: 3px solid white !important;
  box-shadow: 0 2px 8px rgba(22,163,74,0.3) !important;
  width: 20px !important;
  height: 20px !important;
  top: -6px !important;
  transition: all 0.15s ease !important;
}

.stSlider > div > div > div > div[role="slider"]:hover {
  transform: scale(1.15) !important;
  box-shadow: 0 3px 12px rgba(22,163,74,0.4) !important;
}

.stSlider label {
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  color: var(--gray-700) !important;
  letter-spacing: -0.01em !important;
}

.stSelectbox label {
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  color: var(--gray-700) !important;
  letter-spacing: -0.01em !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--green-600) 0%, var(--green-500) 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.7rem 2rem !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  letter-spacing: -0.01em !important;
  box-shadow: 0 4px 16px rgba(22,163,74,.25), 0 1px 3px rgba(22,163,74,.15) !important;
  transition: all 0.25s var(--ease-spring) !important;
  cursor: pointer !important;
}

.stButton > button:hover {
  transform: translateY(-2px) scale(1.01) !important;
  box-shadow: 0 8px 28px rgba(22,163,74,.35), 0 2px 6px rgba(22,163,74,.2) !important;
  background: linear-gradient(135deg, var(--green-700) 0%, var(--green-500) 100%) !important;
}

.stButton > button:active {
  transform: translateY(0px) scale(0.99) !important;
  box-shadow: 0 2px 8px rgba(22,163,74,.25) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--glass-bg-strong) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  border: 1px solid rgba(0,0,0,0.04) !important;
  transition: all 0.2s ease !important;
}

.streamlit-expanderHeader:hover {
  background: rgba(255,255,255,0.9) !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ── DataFrame ── */
.stDataFrame {
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-xs) !important;
  border: 1px solid rgba(0,0,0,0.04) !important;
}

/* ── How It Works Step Cards ── */
.step-card {
  flex: 1;
  min-width: 160px;
  text-align: center;
  padding: 1.5rem 1rem;
  border-radius: var(--radius);
  background: rgba(255,255,255,0.5);
  border: 1px solid rgba(0,0,0,0.03);
  transition: all var(--dur) var(--ease-spring);
}

.step-card:hover {
  background: rgba(255,255,255,0.8);
  transform: translateY(-4px);
  box-shadow: var(--shadow);
  border-color: rgba(22,163,74,0.1);
}

.step-icon {
  font-size: 2.2rem;
  margin-bottom: 0.7rem;
  display: block;
}

.step-title {
  font-weight: 700;
  font-size: 0.92rem;
  color: var(--gray-800);
  letter-spacing: -0.01em;
}

.step-desc {
  font-size: 0.8rem;
  color: var(--gray-400);
  margin-top: 4px;
  line-height: 1.5;
}

/* ── Confidence Meter ── */
.confidence-meter {
  background: var(--glass-bg-strong);
  border: 1px solid rgba(0,0,0,0.04);
  border-radius: var(--radius-lg);
  padding: 2rem 2.4rem;
  margin-top: 1.2rem;
  animation: resultFadeIn 0.7s var(--ease-spring) 0.15s both;
}

/* ── Best Model Banner ── */
.best-model-banner {
  background: linear-gradient(145deg,
    rgba(220,252,231,0.6) 0%,
    rgba(240,253,244,0.8) 50%,
    rgba(255,255,255,0.6) 100%);
  border: 1.5px solid rgba(134,239,172,0.35);
  border-radius: var(--radius-xl);
  padding: 2.5rem 2rem;
  text-align: center;
  box-shadow: var(--shadow-green);
  position: relative;
  overflow: hidden;
  margin-bottom: 2rem;
}

.best-model-banner::before {
  content: '';
  position: absolute;
  top: -100px;
  right: -100px;
  width: 300px;
  height: 300px;
  background: radial-gradient(circle, rgba(74,222,128,0.1) 0%, transparent 70%);
  pointer-events: none;
}

/* ── Active Model Pill ── */
.active-model-card {
  background: var(--glass-bg-strong);
  border: 1px solid rgba(0,0,0,0.04);
  border-radius: var(--radius-lg);
  padding: 1.4rem 1.8rem;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  box-shadow: var(--shadow-sm);
}

/* ── Footer ── */
.footer {
  text-align: center;
  margin-top: 5rem;
  padding: 2rem 1.5rem;
  font-size: 0.78rem;
  color: var(--gray-400);
  border-top: 1px solid rgba(0,0,0,0.05);
  letter-spacing: -0.01em;
}

.footer a {
  color: var(--green-600);
  text-decoration: none;
  font-weight: 500;
}

/* ── Responsive ── */
@media (max-width: 768px) {
  .main .block-container {
    padding: 1.5rem 1.2rem 3rem !important;
  }
  .hero-title { font-size: 2rem !important; }
  .card { padding: 1.5rem 1.4rem; border-radius: var(--radius-lg); }
  .result-card-positive,
  .result-card-negative { padding: 2rem 1.5rem; }
  .result-value { font-size: 2rem; }
}

/* ── Reduced Motion ── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
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
    ax.set_facecolor("#fafdfb")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e5e7eb")
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    ax.tick_params(colors="#9ca3af", labelsize=8.5, length=3, width=0.6)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="600", color="#1f2937",
                      pad=14, fontfamily="sans-serif")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color="#9ca3af", labelpad=8, fontweight="500")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color="#9ca3af", labelpad=8, fontweight="500")


def fig_to_st(fig):
    """Render a matplotlib figure in Streamlit and clean up."""
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    st.pyplot(fig)
    plt.close(fig)


# Sidebar navigation

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 1.2rem;'>
      <div style='
        width: 48px; height: 48px; margin: 0 auto 10px;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        border-radius: 14px; display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; box-shadow: 0 4px 16px rgba(22,163,74,0.25);
      '>🌿</div>
      <div style='font-weight:800; font-size:1.1rem; color:#15803d;
           letter-spacing:-0.02em;'>GreenSense</div>
      <div style='font-size:0.72rem; color:#9ca3af; margin-top:3px;
           font-weight:500; letter-spacing:0.02em;'>ML Insights Platform</div>
    </div>
    <hr style='border:none;border-top:1px solid rgba(0,0,0,0.06);margin:0.6rem 0.8rem 1.2rem;'>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigate",
        ["🏠  Overview", "🔬  Explore Data", "🤖  Train Models", "🎯  Predict", "📊  Insights"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style='border:none;border-top:1px solid rgba(0,0,0,0.06);margin:1.6rem 0.8rem 1.2rem;'>
    <div style='font-size:0.73rem; color:#9ca3af; padding:0 0.6rem; line-height:1.7;'>
      <div style='font-weight:700; color:#374151; font-size:0.68rem;
           text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;'>Dataset</div>
      1,200 synthetic consumer records<br><br>
      <div style='font-weight:700; color:#374151; font-size:0.68rem;
           text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;'>Models</div>
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

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # Overview charts
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class='card'>
          <div class='card-title'>📊 Target Distribution</div>
          <div class='card-subtitle'>Green vs. Non-Green consumer split</div>
        """, unsafe_allow_html=True)
        counts = df["Green_Purchase_Behavior"].value_counts()
        fig, ax = plt.subplots(figsize=(4.5, 3))
        bars = ax.bar(
            ["Non-Green", "Green"],
            [counts[0], counts[1]],
            color=["#d1fae5", "#22c55e"],
            edgecolor="white",
            linewidth=2,
            width=0.5,
            zorder=3,
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#e5e7eb", zorder=0)
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
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.hist(df["Environmental_Awareness"], bins=10, color="#22c55e",
                edgecolor="white", linewidth=1.5, rwidth=0.85, zorder=3, alpha=0.85)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#e5e7eb", zorder=0)
        style_ax(ax, xlabel="Score (1–10)", ylabel="Frequency")
        fig.tight_layout()
        fig_to_st(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # How-it-works workflow steps
    st.markdown("""
    <div class='section-divider'>
      <h3>How it works</h3><div class='line'></div>
    </div>
    <div class='card' style='padding:1.6rem 2rem;'>
      <div style='display:flex; gap:1rem; flex-wrap:wrap;'>
        <div class='step-card'>
          <span class='step-icon'>🔬</span>
          <div class='step-title'>Explore Data</div>
          <div class='step-desc'>Visualize patterns & distributions</div>
        </div>
        <div class='step-card'>
          <span class='step-icon'>🤖</span>
          <div class='step-title'>Train Models</div>
          <div class='step-desc'>Compare 5 ML algorithms</div>
        </div>
        <div class='step-card'>
          <span class='step-icon'>🎯</span>
          <div class='step-title'>Predict</div>
          <div class='step-desc'>Get real-time predictions</div>
        </div>
        <div class='step-card'>
          <span class='step-icon'>📊</span>
          <div class='step-title'>Insights</div>
          <div class='step-desc'>Feature importance & impact</div>
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
    st.dataframe(df.head(50), use_container_width=True, height=280)
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
    fig, axes = plt.subplots(1, len(num_feats), figsize=(15, 3.4))
    for ax, feat in zip(axes, num_feats):
        for val, color in [(0, "#d1fae5"), (1, "#22c55e")]:
            ax.hist(df[df["Green_Purchase_Behavior"] == val][feat],
                    bins=12, alpha=0.7, color=color, edgecolor="white", linewidth=1)
        style_ax(ax, title=feat.replace("_", " "), xlabel="Value", ylabel="")
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#e5e7eb")

    patch0 = mpatches.Patch(color="#d1fae5", label="Non-Green")
    patch1 = mpatches.Patch(color="#22c55e", label="Green")
    fig.legend(handles=[patch0, patch1], loc="upper right", fontsize=8,
               frameon=True, edgecolor="#e5e7eb", facecolor="white", framealpha=0.9)
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
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, feat in zip(axes.flatten(), cat_feats):
        ct = df.groupby(feat)["Green_Purchase_Behavior"].mean().sort_values(ascending=False)
        bars = ax.bar(ct.index, ct.values * 100, color=GREEN_PALETTE[:len(ct)],
                      edgecolor="white", linewidth=1.5, zorder=3, width=0.55)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#e5e7eb", zorder=0)
        ax.axhline(50, color="#f87171", linewidth=0.8, linestyle="--", alpha=0.5)
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
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt=".2f", cmap="Greens",
                linewidths=0.5, linecolor="#f3f4f6", ax=ax,
                annot_kws={"size": 9, "weight": "bold"},
                cbar_kws={"shrink": .65})
    ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold", color="#1f2937", pad=16)
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
    <div class='best-model-banner'>
      <div style='font-size:.68rem;font-weight:700;text-transform:uppercase;
           letter-spacing:.1em;color:#9ca3af;margin-bottom:.6rem;'>🏆 Best Performing Model</div>
      <div style='font-family:"Playfair Display",serif;font-size:2.2rem;color:#15803d;
           font-weight:700;letter-spacing:-0.02em;'>
        {best_name}
      </div>
      <div style='font-size:1.6rem;font-weight:700;color:#16a34a;margin-top:.3rem;
           letter-spacing:-0.01em;'>
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
          <div style='width:190px; font-weight:{"700" if is_best else "500"};
               font-size:.88rem; color:#1f2937; letter-spacing:-0.01em;'>{name}</div>
          {badge}
          <div style='flex:1; background:rgba(0,0,0,0.04); border-radius:99px; height:8px; overflow:hidden;'>
            <div style='width:{bar_w:.1f}%; height:100%;
                 background:linear-gradient(90deg,#16a34a,#4ade80);
                 border-radius:99px; transition: width 0.8s ease;'></div>
          </div>
          <div style='font-weight:700; font-size:.92rem; color:#15803d;
               min-width:56px; text-align:right;'>{acc:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Visual accuracy bar chart and classification report
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("<div class='card'><div class='card-title'>📈 Visual Comparison</div><div class='card-subtitle'>Accuracy by model</div>", unsafe_allow_html=True)
        names = [n.replace(" ", "\n") for n, _ in sorted_models]
        accs  = [info["acc"] for _, info in sorted_models]
        colors = [ACCENT if n.replace("\n", " ") == best_name else "#86efac" for n in names]
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=1.5, zorder=3, width=0.55)
        ax.set_ylim(0.5, 1.0)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#e5e7eb", zorder=0)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                    f"{val:.2%}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#374151")
        style_ax(ax, ylabel="Accuracy")
        fig.tight_layout()
        fig_to_st(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='card'><div class='card-title'>📋 Classification Report</div><div class='card-subtitle'>{best_name}</div>", unsafe_allow_html=True)
        report = trained[best_name]["report"]
        for label, vals in report.items():
            if isinstance(vals, dict):
                name_display = {"0": "Non-Green", "1": "Green"}.get(label, label.title())
                p = vals.get("precision", 0)
                r = vals.get("recall", 0)
                f = vals.get("f1-score", 0)
                st.markdown(f"""
                <div style='padding:10px 0; border-bottom:1px solid rgba(0,0,0,0.04);'>
                  <div style='font-weight:700;font-size:.85rem;color:#374151;
                       letter-spacing:-0.01em;'>{name_display}</div>
                  <div style='display:flex;gap:16px;font-size:.8rem;color:#6b7280;margin-top:4px;'>
                    <span>P: <b style='color:#16a34a;'>{p:.2f}</b></span>
                    <span>R: <b style='color:#16a34a;'>{r:.2f}</b></span>
                    <span>F1: <b style='color:#16a34a;'>{f:.2f}</b></span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Confusion matrix heatmap for the best model
    st.markdown("<div class='card'><div class='card-title'>🔲 Confusion Matrix</div><div class='card-subtitle'>" + best_name + "</div>", unsafe_allow_html=True)
    cm = trained[best_name]["cm"]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Non-Green", "Green"],
                yticklabels=["Non-Green", "Green"],
                linewidths=1, linecolor="#f3f4f6",
                ax=ax, annot_kws={"size": 14, "weight": "bold"})
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
    <div class='active-model-card'>
      <div>
        <div class='card-title' style='margin-bottom:2px;'>⚡ Active Model</div>
        <div style='font-size:.82rem;color:#9ca3af;'>
          Accuracy: <b style='color:#16a34a;'>{trained[best_name]['acc']:.2%}</b>
        </div>
      </div>
      <div style='display:flex;align-items:center;gap:10px;'>
        <span style='font-size:1.05rem;font-weight:700;color:#15803d;
              letter-spacing:-0.01em;'>{best_name}</span>
        <span class='model-badge'>BEST</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Consumer profile input form
    st.markdown("<div class='card'><div class='card-title'>🧾 Consumer Profile</div><div class='card-subtitle'>Configure features below to generate a prediction</div>", unsafe_allow_html=True)

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

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
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

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f"""
            <div class='result-card-positive'>
              <div class='result-label'>Prediction Result</div>
              <div class='result-value' style='color:#15803d;'>🌿 Green Consumer</div>
              <div class='result-desc'>
                This consumer profile indicates a strong likelihood of eco-friendly purchasing behavior.
              </div>
              <span class='confidence-tag {conf_cls}'>{conf_label} · {prob:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card-negative'>
              <div class='result-label'>Prediction Result</div>
              <div class='result-value' style='color:#dc2626;'>❌ Non-Green Consumer</div>
              <div class='result-desc'>
                This consumer profile suggests limited inclination toward eco-friendly purchasing.
              </div>
              <span class='confidence-tag {conf_cls}'>{conf_label} · {prob:.0%}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='confidence-meter'>
          <div class='card-title' style='margin-bottom:12px;'>📊 Prediction Confidence</div>
        """, unsafe_allow_html=True)
        st.progress(float(prob))
        st.markdown(f"""
          <div style='display:flex;justify-content:space-between;font-size:.75rem;
               color:#9ca3af;margin-top:6px;font-weight:500;'>
            <span>0%</span><span>50%</span><span>100%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


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

    st.markdown("<div class='card'><div class='card-title'>🌲 Feature Importance</div><div class='card-subtitle'>Random Forest — which factors drive green behavior most?</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
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
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="#e5e7eb", zorder=0)
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
              <div style='font-size:1.8rem;'>{icon}</div>
              <div style='font-weight:700;font-size:.92rem;color:#1f2937;margin:.6rem 0 .2rem;
                   letter-spacing:-0.01em;'>
                {feat.replace('_', ' ')}
              </div>
              <div class='metric-val' style='font-size:1.6rem;'>{val:.3f}</div>
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
        st.markdown("<div class='card'><div class='card-title'>🌱 Awareness vs Social Influence</div><div class='card-subtitle'>Colored by consumer behavior</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        colors = df["Green_Purchase_Behavior"].map({0: "#d1fae5", 1: "#16a34a"})
        ax.scatter(df["Environmental_Awareness"], df["Social_Influence"],
                   c=colors, alpha=0.5, s=20, edgecolors="white", linewidths=0.4, zorder=3)
        ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="#e5e7eb", zorder=0)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#e5e7eb", zorder=0)
        style_ax(ax, xlabel="Environmental Awareness", ylabel="Social Influence")
        p0 = mpatches.Patch(color="#d1fae5", label="Non-Green")
        p1 = mpatches.Patch(color="#16a34a", label="Green")
        ax.legend(handles=[p0, p1], fontsize=8, frameon=True, edgecolor="#e5e7eb",
                  facecolor="white", framealpha=0.9)
        fig.tight_layout()
        fig_to_st(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'><div class='card-title'>💲 Price Sensitivity by Behavior</div><div class='card-subtitle'>Comparing green vs non-green consumers</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        g0 = df[df["Green_Purchase_Behavior"] == 0]["Price_Sensitivity"]
        g1 = df[df["Green_Purchase_Behavior"] == 1]["Price_Sensitivity"]
        bp = ax.boxplot([g0, g1], labels=["Non-Green", "Green"],
                        patch_artist=True, notch=True, widths=0.45,
                        medianprops=dict(color="white", linewidth=2))
        colors_box = ["#d1fae5", "#22c55e"]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#e5e7eb", zorder=0)
        style_ax(ax, ylabel="Price Sensitivity Score")
        fig.tight_layout()
        fig_to_st(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
  🌿 <b>GreenSense</b> · Machine Learning Platform &nbsp;·&nbsp;
  Built with Streamlit &amp; scikit-learn &nbsp;·&nbsp;
  Data is synthetic &amp; for demonstration purposes
</div>
""", unsafe_allow_html=True)
