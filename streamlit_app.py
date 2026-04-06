"""
============================================================
    GREEN CONSUMER BEHAVIOR PREDICTION - STREAMLIT WEB APP
    Premium Streamlit-native UI redesign
    Run with: streamlit run streamlit_app.py
============================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Green Consumer Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# THEME + UI HELPERS
# ---------------------------------------------------------
def inject_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Urbanist:wght@600;700;800&display=swap');

:root {
    --bg-0: #020617;
    --bg-1: #0b1220;
    --bg-2: #0f172a;
    --ink-0: #e2e8f0;
    --ink-1: #94a3b8;
    --line: rgba(148, 163, 184, 0.22);
    --glass: rgba(15, 23, 42, 0.52);
    --glass-strong: rgba(15, 23, 42, 0.72);
    --primary: #38bdf8;
    --primary-2: #60a5fa;
    --good: #22c55e;
    --warn: #f59e0b;
    --bad: #f43f5e;
    --shadow-soft: 0 10px 35px rgba(2, 6, 23, 0.42);
    --shadow-glow: 0 0 0 1px rgba(56, 189, 248, 0.18), 0 14px 30px rgba(56, 189, 248, 0.18);
    --rad-xl: 24px;
    --rad-lg: 18px;
    --rad-md: 14px;
    --ease-out-premium: cubic-bezier(.22, .83, .25, .99);
    --dur-fast: .24s;
    --dur-med: .42s;
}

html, body, [class*="css"] {
    font-family: Inter, "SF Pro Text", "Segoe UI", sans-serif;
}

.stApp {
    color: var(--ink-0);
    background:
      radial-gradient(1200px 700px at 90% -10%, rgba(56, 189, 248, 0.18), transparent 55%),
      radial-gradient(900px 500px at 10% 10%, rgba(96, 165, 250, 0.12), transparent 50%),
      linear-gradient(160deg, var(--bg-0) 0%, var(--bg-1) 40%, var(--bg-2) 100%);
}

.block-container {
    max-width: 1180px;
    padding-top: .85rem;
    padding-bottom: 4.8rem;
}

.glass-nav {
    position: sticky;
    top: .85rem;
    z-index: 100;
    margin-bottom: 1.25rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: rgba(2, 6, 23, 0.48);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: var(--shadow-soft);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: .62rem .95rem;
    transition: background var(--dur-med) var(--ease-out-premium), border-color var(--dur-med) var(--ease-out-premium), box-shadow var(--dur-med) var(--ease-out-premium), transform var(--dur-med) var(--ease-out-premium);
}

.glass-nav.scrolled {
    background: rgba(2, 6, 23, 0.74);
    border-color: rgba(125, 211, 252, .32);
    box-shadow: 0 16px 34px rgba(2, 6, 23, .5), 0 0 0 1px rgba(56, 189, 248, .15);
    transform: translateY(-1px);
}

.brand-wrap {
    display: flex;
    align-items: center;
    gap: .75rem;
}

.brand-dot {
    width: 30px;
    height: 30px;
    border-radius: 999px;
    background: linear-gradient(145deg, var(--primary), #22d3ee);
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.45);
}

.brand-title {
    font-family: Urbanist, Inter, sans-serif;
    font-weight: 700;
    font-size: .98rem;
    letter-spacing: .01em;
    color: #e2e8f0;
}

.brand-sub {
    color: var(--ink-1);
    font-size: .76rem;
}

.nav-pills {
    display: flex;
    align-items: center;
    gap: .45rem;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.pill {
    border: 1px solid var(--line);
    background: rgba(15, 23, 42, 0.55);
    color: #cbd5e1;
    border-radius: 999px;
    font-size: .72rem;
    padding: .35rem .8rem;
    transition: background var(--dur-fast) var(--ease-out-premium), border-color var(--dur-fast) var(--ease-out-premium), transform var(--dur-fast) var(--ease-out-premium);
}

.pill:hover {
    transform: translateY(-1px);
    border-color: rgba(125, 211, 252, .45);
    background: rgba(30, 41, 59, .72);
}

.hero {
    margin: 1.05rem 0 2.5rem 0;
    border: 1px solid rgba(96, 165, 250, .25);
    border-radius: var(--rad-xl);
    background:
      linear-gradient(135deg, rgba(7, 12, 22, .78), rgba(12, 21, 38, .58)),
      radial-gradient(120% 120% at 80% 0%, rgba(56, 189, 248, .22), transparent 55%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: var(--shadow-glow);
    overflow: hidden;
    position: relative;
    transition: transform .5s var(--ease-out-premium);
}

.hero::before {
    content: "";
    position: absolute;
    inset: -40% auto auto -10%;
    width: 360px;
    height: 360px;
    border-radius: 50%;
    background: radial-gradient(circle at center, rgba(125, 211, 252, .22), transparent 65%);
    filter: blur(8px);
    pointer-events: none;
    animation: drift 8s ease-in-out infinite;
}

.hero-body {
    position: relative;
    z-index: 2;
    padding: 3.1rem clamp(1.25rem, 4vw, 3.15rem);
}

.kicker {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    border: 1px solid rgba(148, 163, 184, .35);
    border-radius: 999px;
    padding: .35rem .85rem;
    font-size: .76rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #cbd5e1;
    background: rgba(15, 23, 42, .45);
}

.hero h1 {
    font-family: Urbanist, Inter, sans-serif;
    font-size: clamp(2rem, 5.6vw, 3.6rem);
    line-height: 1.05;
    letter-spacing: -.04em;
    margin: .86rem 0 .82rem 0;
    color: #f8fafc;
}

.hero p {
    margin: 0;
    color: #cbd5e1;
    font-size: clamp(1rem, 1.25vw, 1.15rem);
    line-height: 1.72;
    max-width: 840px;
}

.stack { margin-top: 2.6rem; }

.section-title {
    margin: 0 0 1.15rem 0;
    font-family: Urbanist, Inter, sans-serif;
    font-size: clamp(1.2rem, 1.2vw, 1.36rem);
    letter-spacing: -.01em;
    color: #f1f5f9;
}

.glass-card {
    border: 1px solid var(--line);
    border-radius: var(--rad-lg);
    background: var(--glass);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: var(--shadow-soft);
    padding: 1.3rem;
    transition: transform var(--dur-med) var(--ease-out-premium), box-shadow var(--dur-med) ease, border-color var(--dur-med) ease;
}

.glass-card:hover {
    transform: translateY(-4px) scale(1.006);
    box-shadow: 0 22px 40px rgba(2, 6, 23, .45);
    border-color: rgba(96, 165, 250, .42);
}

.card-title {
    color: #e2e8f0;
    font-size: 1.02rem;
    font-weight: 600;
    margin-bottom: .45rem;
}

.card-text {
    color: #94a3b8;
    line-height: 1.68;
    font-size: .96rem;
}

.metric-grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(4, minmax(0,1fr));
}

.metric {
    border: 1px solid var(--line);
    border-radius: var(--rad-md);
    background: rgba(2, 6, 23, .45);
    padding: 1.05rem;
    transition: transform var(--dur-fast) var(--ease-out-premium), box-shadow var(--dur-fast) ease, border-color var(--dur-fast) ease;
}

.metric:hover {
    transform: translateY(-3px);
    border-color: rgba(56, 189, 248, .4);
    box-shadow: 0 12px 30px rgba(2, 6, 23, .45);
}

.metric-kicker {
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-size: .68rem;
    font-weight: 600;
}

.metric-value {
    margin-top: .25rem;
    color: #f8fafc;
    font-size: clamp(1.4rem, 2.45vw, 2.04rem);
    font-weight: 700;
    letter-spacing: -.02em;
}

.metric-label {
    color: #94a3b8;
    font-size: .81rem;
    margin-top: .18rem;
}

.soft-divider {
    margin: 1.75rem 0;
    height: 1px;
    background: linear-gradient(90deg, rgba(148,163,184,.05), rgba(148,163,184,.35), rgba(148,163,184,.05));
}

.plot-shell {
    border: 1px solid var(--line);
    border-radius: var(--rad-lg);
    background: var(--glass-strong);
    padding: .82rem;
    box-shadow: var(--shadow-soft);
    transition: transform var(--dur-med) var(--ease-out-premium), box-shadow var(--dur-med) ease, border-color var(--dur-med) ease;
}

.plot-shell:hover {
    transform: translateY(-2px);
    border-color: rgba(96, 165, 250, .45);
    box-shadow: 0 20px 34px rgba(2, 6, 23, .45);
}

.reveal {
    opacity: 0;
    transform: translateY(16px);
    transition: opacity .62s var(--ease-out-premium), transform .62s var(--ease-out-premium);
}

.reveal.in-view {
    opacity: 1;
    transform: translateY(0px);
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(2,6,23,.92), rgba(11,18,32,.9));
    border-right: 1px solid var(--line);
}

div[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem;
}

.sidebar-box {
    border: 1px solid var(--line);
    border-radius: 16px;
    background: rgba(15, 23, 42, .5);
    padding: .95rem;
    margin-bottom: .95rem;
    transition: border-color var(--dur-fast) var(--ease-out-premium), transform var(--dur-fast) var(--ease-out-premium), background var(--dur-fast) var(--ease-out-premium);
}

.sidebar-box:hover {
    transform: translateY(-1px);
    border-color: rgba(125, 211, 252, .34);
    background: rgba(30, 41, 59, .48);
}

.sidebar-box h4 {
    margin: 0 0 .35rem 0;
    color: #f1f5f9;
    font-size: .97rem;
}

.sidebar-box p {
    margin: 0;
    color: #94a3b8;
    line-height: 1.6;
    font-size: .84rem;
}

.stButton > button,
.stForm button[kind="primary"] {
    border-radius: 999px !important;
    border: 1px solid rgba(56, 189, 248, .35) !important;
    background: linear-gradient(135deg, rgba(14,165,233,.26), rgba(59,130,246,.2)) !important;
    color: #f8fafc !important;
    height: 2.9rem !important;
    font-weight: 600 !important;
    box-shadow: 0 8px 20px rgba(14, 165, 233, .25) !important;
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease !important;
}

.stButton > button:hover,
.stForm button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.01) !important;
    border-color: rgba(125, 211, 252, .8) !important;
    box-shadow: 0 14px 28px rgba(14, 165, 233, .34) !important;
}

.stSelectbox label,
.stSlider label,
.stNumberInput label,
.stRadio label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: rgba(15, 23, 42, .62) !important;
    border: 1px solid var(--line) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
}

@media (max-width: 1080px) {
    .metric-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
    .glass-nav { border-radius: 18px; }
}

@media (max-width: 720px) {
    .metric-grid { grid-template-columns: 1fr; }
    .hero-body { padding: 1.6rem 1rem; }
    .glass-nav {
        flex-direction: column;
        align-items: flex-start;
        gap: .6rem;
    }
    .nav-pills {
        width: 100%;
        justify-content: flex-start;
    }
}

@media (prefers-reduced-motion: reduce) {
    .reveal,
    .glass-card,
    .metric,
    .plot-shell,
    .pill,
    .glass-nav,
    .hero {
        animation: none !important;
        transition: none !important;
        transform: none !important;
    }
}

@keyframes drift {
    0%, 100% { transform: translate(0px, 0px); }
    50% { transform: translate(12px, 12px); }
}
</style>
""",
        unsafe_allow_html=True,
    )

    # Streamlit-compatible scroll reveal and subtle parallax via parent document observer.
    components.html(
        """
<script>
(function () {
  const pdoc = window.parent && window.parent.document ? window.parent.document : document;
  function mount() {
    const els = pdoc.querySelectorAll('.reveal');
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add('in-view');
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -30px 0px' });

    els.forEach(el => observer.observe(el));

        const hero = pdoc.querySelector('.hero');
        const nav = pdoc.querySelector('.glass-nav');
    const onScroll = () => {
            const sy = window.parent.scrollY || 0;
            if (nav) {
                if (sy > 24) nav.classList.add('scrolled');
                else nav.classList.remove('scrolled');
            }
      if (!hero) return;
            const y = Math.min(sy, 240);
      hero.style.transform = `translateY(${y * 0.03}px)`;
    };

    window.parent.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  setTimeout(mount, 60);
})();
</script>
""",
        height=0,
        width=0,
    )


def glass_nav(best_model_name: str, best_accuracy: float) -> None:
    st.markdown(
        f"""
<div class="glass-nav reveal">
  <div class="brand-wrap">
    <div class="brand-dot"></div>
    <div>
      <div class="brand-title">Green Consumer Intelligence</div>
      <div class="brand-sub">Premium ML Analytics Dashboard</div>
    </div>
  </div>
  <div class="nav-pills">
    <span class="pill">Streamlit Native</span>
    <span class="pill">Best: {best_model_name}</span>
    <span class="pill">Accuracy {best_accuracy:.1f}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def hero_section() -> None:
    st.markdown(
        """
<section class="hero reveal">
  <div class="hero-body">
    <span class="kicker">Sustainable Analytics Platform</span>
    <h1>Predict Eco-First Buyers<br/>With Precision And Clarity.</h1>
    <p>
      A modern, high-end interface for exploring green consumer behavior using machine learning.
      Built for smooth workflows, strong visual hierarchy, and production-grade user experience.
    </p>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_plot(fig) -> None:
    st.markdown('<div class="plot-shell reveal">', unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# DATA + MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_data():
    np.random.seed(42)
    n_rows = 500
    scaled_models = ["Logistic Regression"]

    age = np.random.randint(18, 70, n_rows)
    income = np.random.randint(20000, 150000, n_rows)
    education_level = np.random.choice([0, 1, 2, 3], n_rows, p=[0.05, 0.30, 0.40, 0.25])
    environmental_concern = np.random.randint(1, 11, n_rows)
    social_influence = np.random.randint(1, 11, n_rows)
    eco_awareness = np.random.randint(1, 11, n_rows)
    past_green_purchases = np.random.randint(0, 20, n_rows)
    price_sensitivity = np.random.randint(1, 11, n_rows)
    marketing_exposure = np.random.randint(1, 11, n_rows)
    gender = np.random.choice([0, 1], n_rows)
    location = np.random.choice([0, 1, 2], n_rows, p=[0.20, 0.35, 0.45])

    score = (
        0.25 * (environmental_concern / 10)
        + 0.20 * (eco_awareness / 10)
        + 0.15 * (social_influence / 10)
        + 0.15 * (past_green_purchases / 20)
        + 0.10 * (income / 150000)
        + 0.10 * (education_level / 3)
        + 0.05 * (1 - price_sensitivity / 10)
        + np.random.normal(0, 0.05, n_rows)
    )
    green_consumer = (score > 0.45).astype(int)

    df = pd.DataFrame(
        {
            "Age": age,
            "Income": income,
            "Education_Level": education_level,
            "Environmental_Concern": environmental_concern,
            "Social_Influence": social_influence,
            "Eco_Awareness": eco_awareness,
            "Past_Green_Purchases": past_green_purchases,
            "Price_Sensitivity": price_sensitivity,
            "Marketing_Exposure": marketing_exposure,
            "Gender": gender,
            "Location": location,
            "Green_Consumer": green_consumer,
        }
    )

    x_data = df.drop("Green_Consumer", axis=1)
    y_data = df["Green_Consumer"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
    }

    trained = {}
    for name, model in models.items():
        if name in scaled_models:
            model.fit(x_train_sc, y_train)
            y_pred = model.predict(x_test_sc)
        else:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

        trained[name] = {
            "model": model,
            "acc": accuracy_score(y_test, y_pred),
            "y_pred": y_pred,
        }

    best_model_name = max(trained, key=lambda k: trained[k]["acc"])
    return df, x_data, y_data, x_train, x_test, y_train, y_test, scaler, trained, scaled_models, best_model_name


# ---------------------------------------------------------
# APP BOOTSTRAP
# ---------------------------------------------------------
inject_theme()
plt.style.use("dark_background")

with st.spinner("Training models..."):
    (
        df,
        x_data,
        y_data,
        x_train,
        x_test,
        y_train,
        y_test,
        scaler,
        trained,
        scaled_models,
        best_model_name,
    ) = load_model_and_data()

best_accuracy = trained[best_model_name]["acc"] * 100

glass_nav(best_model_name, best_accuracy)
hero_section()


# ---------------------------------------------------------
# SIDEBAR NAV
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
<div class="sidebar-box">
  <h4>Navigation</h4>
  <p>Move through insights, visual analytics, model benchmarking, and live prediction.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    tab_choice = st.radio(
        "Go to section",
        [
            "Overview",
            "EDA and Visualizations",
            "Model Comparison",
            "Live Prediction",
            "Project Info",
        ],
        label_visibility="visible",
    )

    st.markdown(
        f"""
<div class="sidebar-box">
  <h4>Project Snapshot</h4>
  <p>
    Records: 500<br/>
    Features: 11<br/>
    Models: 4 classifiers<br/>
    Best Model: {best_model_name}<br/>
    Accuracy: {best_accuracy:.1f}%
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------
if tab_choice == "Overview":
    st.markdown('<h2 class="section-title reveal">Problem Statement</h2>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="glass-card reveal">
  <div class="card-text">
    Predict whether a user is likely to purchase eco-friendly products using socio-demographic and behavioral signals.
    <br/><br/>
    Target classes: <strong>1 = Green Consumer</strong>, <strong>0 = Non-Green Consumer</strong>.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="stack"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title reveal">Core Objectives</h2>', unsafe_allow_html=True)

    objective_cards = [
        "Build realistic synthetic consumer data",
        "Explore behavioral patterns through EDA",
        "Train and compare classification models",
        "Evaluate accuracy and ROC-AUC",
        "Identify key drivers of green buying",
        "Enable interactive prediction workflow",
    ]

    cols = st.columns(3, gap="large")
    for idx, text in enumerate(objective_cards):
        with cols[idx % 3]:
            st.markdown(
                f"""
<div class="glass-card reveal">
  <div class="card-title">Objective {idx + 1:02d}</div>
  <div class="card-text">{text}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="stack"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title reveal">Quick Metrics</h2>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="metric-grid reveal">
  <div class="metric">
    <div class="metric-kicker">Total Records</div>
    <div class="metric-value">{len(df):,}</div>
    <div class="metric-label">Consumer profiles</div>
  </div>
  <div class="metric">
    <div class="metric-kicker">Input Features</div>
    <div class="metric-value">11</div>
    <div class="metric-label">Behavior + demographic</div>
  </div>
  <div class="metric">
    <div class="metric-kicker">Green Class</div>
    <div class="metric-value">{df['Green_Consumer'].sum()}</div>
    <div class="metric-label">Positive outcomes</div>
  </div>
  <div class="metric">
    <div class="metric-kicker">Non-Green Class</div>
    <div class="metric-value">{(df['Green_Consumer'] == 0).sum()}</div>
    <div class="metric-label">Negative outcomes</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title reveal">Dataset Preview</h2>', unsafe_allow_html=True)
    st.dataframe(df.head(12), use_container_width=True)


# ---------------------------------------------------------
# EDA
# ---------------------------------------------------------
elif tab_choice == "EDA and Visualizations":
    st.markdown('<h2 class="section-title reveal">Exploratory Visual Analytics</h2>', unsafe_allow_html=True)

    viz = st.selectbox(
        "Choose Visualization",
        [
            "Target Distribution",
            "Feature Distributions",
            "Correlation Heatmap",
            "Boxplots by Consumer Type",
            "Income vs Environmental Concern",
        ],
    )

    if viz == "Target Distribution":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        counts = df["Green_Consumer"].value_counts()
        axes[0].pie(
            counts,
            labels=["Green", "Non-Green"],
            autopct="%1.1f%%",
            colors=["#22c55e", "#f43f5e"],
            wedgeprops={"edgecolor": "#0f172a", "linewidth": 2},
            textprops={"color": "#e2e8f0"},
        )
        axes[0].set_title("Class Proportion", color="#e2e8f0")

        axes[1].bar(["Non-Green", "Green"], counts.values, color=["#f43f5e", "#22c55e"])
        axes[1].set_title("Class Count", color="#e2e8f0")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(colors="#cbd5e1")
        render_plot(fig)

    elif viz == "Feature Distributions":
        feature = st.selectbox("Select Feature", x_data.columns.tolist())
        fig, ax = plt.subplots(figsize=(9, 4.3))
        ax.hist(
            df[df["Green_Consumer"] == 0][feature],
            bins=16,
            alpha=0.55,
            color="#f43f5e",
            label="Non-Green",
        )
        ax.hist(
            df[df["Green_Consumer"] == 1][feature],
            bins=16,
            alpha=0.55,
            color="#22c55e",
            label="Green",
        )
        ax.set_title(f"Distribution of {feature}")
        ax.legend()
        ax.grid(alpha=0.18)
        render_plot(fig)

    elif viz == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(11, 8))
        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="icefire",
            mask=mask,
            linewidths=0.6,
            ax=ax,
            vmin=-1,
            vmax=1,
            annot_kws={"size": 8},
        )
        ax.set_title("Correlation Heatmap")
        render_plot(fig)

    elif viz == "Boxplots by Consumer Type":
        feature = st.selectbox("Select Feature", x_data.columns.tolist(), key="box_feature")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bp = ax.boxplot(
            [df[df["Green_Consumer"] == 0][feature], df[df["Green_Consumer"] == 1][feature]],
            labels=["Non-Green", "Green"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("#f43f5e")
        bp["boxes"][1].set_facecolor("#22c55e")
        ax.set_title(f"{feature} by Consumer Type")
        ax.grid(alpha=0.2)
        render_plot(fig)

    else:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for label, color, marker in [(0, "#f43f5e", "x"), (1, "#22c55e", "o")]:
            subset = df[df["Green_Consumer"] == label]
            ax.scatter(
                subset["Income"],
                subset["Environmental_Concern"],
                c=color,
                label="Green" if label else "Non-Green",
                alpha=0.55,
                s=42,
                marker=marker,
            )
        ax.set_xlabel("Income")
        ax.set_ylabel("Environmental Concern")
        ax.set_title("Income vs Environmental Concern")
        ax.grid(alpha=0.24)
        ax.legend()
        render_plot(fig)

    st.markdown('<div class="stack"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title reveal">Summary Statistics</h2>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)


# ---------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------
elif tab_choice == "Model Comparison":
    st.markdown('<h2 class="section-title reveal">Model Benchmarks</h2>', unsafe_allow_html=True)

    model_names = list(trained.keys())
    accuracies = [trained[m]["acc"] * 100 for m in model_names]

    metric_html = []
    for name, acc in zip(model_names, accuracies):
        metric_html.append(
            f"""
<div class="metric">
  <div class="metric-kicker">{name}</div>
  <div class="metric-value">{acc:.1f}%</div>
  <div class="metric-label">Test Accuracy</div>
</div>
"""
        )

    st.markdown(f'<div class="metric-grid reveal">{"".join(metric_html)}</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    bars = ax.bar(model_names, accuracies, color=["#22c55e", "#0ea5e9", "#f59e0b", "#f43f5e"], width=0.55)
    ax.set_ylim(60, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison")
    ax.grid(alpha=0.22, axis="y")
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"{acc:.1f}%",
            ha="center",
            fontsize=9,
            color="#e2e8f0",
        )
    render_plot(fig)

    st.markdown('<div class="stack"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title reveal">Feature Importance (Random Forest)</h2>', unsafe_allow_html=True)
    rf_model = trained["Random Forest"]["model"]
    feat_imp = pd.Series(rf_model.feature_importances_, index=x_data.columns).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    feat_imp.plot(kind="barh", ax=ax, color=sns.color_palette("mako", len(feat_imp)))
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Feature Contributions")
    render_plot(fig)

    st.markdown('<div class="stack"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title reveal">Confusion Matrix and ROC</h2>', unsafe_allow_html=True)

    selected_model = st.selectbox("Select Model", model_names)
    y_pred_sel = trained[selected_model]["y_pred"]

    c1, c2 = st.columns([1, 1.25], gap="large")

    with c1:
        cm = confusion_matrix(y_test, y_pred_sel)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="crest",
            xticklabels=["Non-Green", "Green"],
            yticklabels=["Non-Green", "Green"],
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(f"{selected_model} Confusion Matrix")
        render_plot(fig_cm)

    with c2:
        fig_roc, ax_roc = plt.subplots(figsize=(6.2, 4.3))
        ax_roc.plot([0, 1], [0, 1], "--", color="#94a3b8", alpha=0.7, label="Random (AUC=0.50)")
        palette = ["#22c55e", "#38bdf8", "#f59e0b", "#f43f5e"]

        for name, color in zip(model_names, palette):
            model = trained[name]["model"]
            if name in scaled_models:
                y_prob = model.predict_proba(scaler.transform(x_test))[:, 1]
            else:
                y_prob = model.predict_proba(x_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves")
        ax_roc.legend(fontsize=8)
        ax_roc.grid(alpha=0.2)
        render_plot(fig_roc)


# ---------------------------------------------------------
# LIVE PREDICTION
# ---------------------------------------------------------
elif tab_choice == "Live Prediction":
    st.markdown('<h2 class="section-title reveal">Live Green Consumer Prediction</h2>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="glass-card reveal">
  <div class="card-text">Use this panel to generate real-time predictions with smooth, production-grade UX.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    selected_model = st.selectbox("Choose Model", list(trained.keys()))

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            age_in = st.slider("Age", 18, 70, 28)
            income_in = st.number_input("Annual Income", 20000, 150000, 75000, step=5000)
            gender_in = st.selectbox("Gender", ["Female (0)", "Male (1)"])
            location_in = st.selectbox("Location", ["Rural (0)", "Suburban (1)", "Urban (2)"])
            edu_in = st.selectbox(
                "Education Level",
                ["No Formal (0)", "High School (1)", "Graduate (2)", "Post-Graduate (3)"],
            )

        with c2:
            env_concern = st.slider("Environmental Concern", 1, 10, 7)
            eco_aware = st.slider("Eco Awareness", 1, 10, 7)
            past_buy = st.slider("Past Green Purchases", 0, 19, 10)
            soc_inf = st.slider("Social Influence", 1, 10, 6)

        with c3:
            price_sens = st.slider("Price Sensitivity", 1, 10, 4)
            mkt_exp = st.slider("Marketing Exposure", 1, 10, 6)

        submitted = st.form_submit_button("Predict Now", use_container_width=True)

    if submitted:
        gender_val = int(gender_in.split("(")[1].replace(")", ""))
        location_val = int(location_in.split("(")[1].replace(")", ""))
        edu_val = int(edu_in.split("(")[1].replace(")", ""))

        user_data = pd.DataFrame(
            [
                {
                    "Age": age_in,
                    "Income": income_in,
                    "Education_Level": edu_val,
                    "Environmental_Concern": env_concern,
                    "Social_Influence": soc_inf,
                    "Eco_Awareness": eco_aware,
                    "Past_Green_Purchases": past_buy,
                    "Price_Sensitivity": price_sens,
                    "Marketing_Exposure": mkt_exp,
                    "Gender": gender_val,
                    "Location": location_val,
                }
            ]
        )[x_data.columns]

        model = trained[selected_model]["model"]
        if selected_model in scaled_models:
            user_sc = scaler.transform(user_data)
            pred = model.predict(user_sc)[0]
            prob = model.predict_proba(user_sc)[0][1] * 100
        else:
            pred = model.predict(user_data)[0]
            prob = model.predict_proba(user_data)[0][1] * 100

        confidence = "High" if prob > 80 else "Moderate" if prob >= 60 else "Low"

        col_res, col_stats = st.columns([1.7, 1], gap="large")

        with col_res:
            if pred == 1:
                st.markdown(
                    f"""
<div class="glass-card reveal" style="border-color: rgba(34,197,94,.45);">
  <div class="card-title">Prediction: Green Consumer</div>
  <div class="card-text">Likely to prefer eco-friendly products.</div>
  <div class="soft-divider"></div>
  <div class="card-text">Confidence level: <strong>{confidence}</strong></div>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
<div class="glass-card reveal" style="border-color: rgba(244,63,94,.45);">
  <div class="card-title">Prediction: Non-Green Consumer</div>
  <div class="card-text">Less likely to prefer eco-friendly products.</div>
  <div class="soft-divider"></div>
  <div class="card-text">Confidence level: <strong>{confidence}</strong></div>
</div>
""",
                    unsafe_allow_html=True,
                )

        with col_stats:
            st.markdown(
                f"""
<div class="metric-grid reveal" style="grid-template-columns: 1fr;">
  <div class="metric">
    <div class="metric-kicker">Green Probability</div>
    <div class="metric-value">{prob:.1f}%</div>
    <div class="metric-label">Positive class</div>
  </div>
  <div class="metric">
    <div class="metric-kicker">Non-Green Probability</div>
    <div class="metric-value">{100 - prob:.1f}%</div>
    <div class="metric-label">Negative class</div>
  </div>
  <div class="metric">
    <div class="metric-kicker">Model</div>
    <div class="metric-value" style="font-size:1.1rem;">{selected_model}</div>
    <div class="metric-label">Selected classifier</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        fig, ax = plt.subplots(figsize=(7, 1.7))
        ax.barh([""], [prob], color="#22c55e", height=0.4)
        ax.barh([""], [100 - prob], left=[prob], color="#f43f5e", height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability")
        ax.axvline(50, color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_title("Green vs Non-Green Probability")
        render_plot(fig)


# ---------------------------------------------------------
# PROJECT INFO
# ---------------------------------------------------------
else:
    st.markdown('<h2 class="section-title reveal">Project Architecture</h2>', unsafe_allow_html=True)
    st.code(
        f"""
[Data Generation] -> [EDA] -> [Preprocessing] -> [Model Training]
                                        |
                  +---------------------+----------------------+
                  | Logistic Regression | Decision Tree        |
                  | Random Forest       | Gradient Boosting    |
                  +---------------------+----------------------+
                                        |
                                  [Evaluation]
                             Accuracy / ROC-AUC / Insights
                                        |
                               [Streamlit Premium UI]
Best model: {best_model_name}
""",
        language="text",
    )

    st.markdown('<div class="stack"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title reveal">Future Scope</h2>', unsafe_allow_html=True)

    next_steps = [
        "Use real purchase and survey datasets for stronger generalization",
        "Add explainability with SHAP for transparent predictions",
        "Extend to multi-class green behavior scoring",
        "Serve predictions through API endpoints for integration",
        "Track longitudinal behavior trends with time-series modeling",
    ]

    cols = st.columns(2, gap="large")
    for idx, item in enumerate(next_steps):
        with cols[idx % 2]:
            st.markdown(
                f"""
<div class="glass-card reveal">
  <div class="card-title">Roadmap {idx + 1:02d}</div>
  <div class="card-text">{item}</div>
</div>
""",
                unsafe_allow_html=True,
            )
