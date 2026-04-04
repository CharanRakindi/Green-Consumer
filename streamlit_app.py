"""
============================================================
    GREEN CONSUMER BEHAVIOR PREDICTION - STREAMLIT WEB APP
    Run with: streamlit run streamlit_app.py
============================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Green Consumer Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg-color: #fbfbfd;
        --surface-color: #ffffff;
        --text-primary: #1d1d1f;
        --text-secondary: #86868b;
        --border-color: #d2d2d7;
        --accent-green: #2e7d32;
        --radius-lg: 18px;
        --radius-md: 12px;
        --shadow-sm: 0 4px 6px rgba(0, 0, 0, 0.02);
        --shadow-hover: 0 10px 20px rgba(0, 0, 0, 0.04);
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", "Roboto", "Helvetica Neue", sans-serif;
        color: var(--text-primary);
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background-color: var(--bg-color);
    }

    .block-container {
        padding-top: 3rem;
        padding-bottom: 4rem;
        max-width: 1000px;
    }

    .main-header {
        background-color: transparent;
        padding: 1rem 0 3rem 0;
        color: var(--text-primary);
        text-align: center;
    }

    .hero-row {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        background: #f5f5f7;
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .main-header h1 {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        margin: 0 !important;
        color: var(--text-primary);
    }

    .hero-row p {
        color: var(--text-secondary);
        font-size: 1.15rem;
        max-width: 650px;
        line-height: 1.5;
        margin: 1rem auto 0 auto;
    }

    .section-card, .info-card, .metric-card, .prediction-card, .callout {
        background: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .section-card {
        padding: 2rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.2rem;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-spacing {
        margin-top: 3.5rem;
    }

    .subtle-text {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.6;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.2rem;
        margin-top: 1rem;
    }

    .metric-card {
        padding: 1.8rem;
        text-align: center;
        border-top: none;
    }

    .metric-card:hover, .info-card:hover, .prediction-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }

    .metric-kicker {
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }

    .metric-value {
        color: var(--text-primary);
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.4rem;
    }

    .info-card {
        padding: 1.8rem;
        margin-bottom: 1rem;
    }

    .info-card h4 {
        margin: 0 0 0.8rem 0;
        color: var(--text-primary);
        font-size: 1.15rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    .info-card ul, .info-card p {
        margin: 0;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    .info-card ul {
        margin-left: 1.5rem;
    }

    .info-card.green {
        border-top: 3px solid var(--accent-green);
        border-left: none;
        background: var(--surface-color);
    }

    .info-card.gray {
        border-top: 3px solid var(--text-secondary);
        border-left: none;
        background: var(--surface-color);
    }
    
    .info-card.green h4 {
        color: var(--accent-green);
    }

    .prediction-card {
        padding: 2rem;
        border: 1px solid var(--border-color);
        background: var(--surface-color);
    }

    .prediction-banner-green {
        background: #f5fbf5;
        border: 1px solid #d4ecd4;
        border-radius: var(--radius-md);
        padding: 2rem;
        text-align: center;
        color: var(--text-primary);
    }

    .prediction-banner-red {
        background: #fdf5f5;
        border: 1px solid #ecd4d4;
        border-radius: var(--radius-md);
        padding: 2rem;
        text-align: center;
        color: var(--text-primary);
    }

    .prediction-title {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.6rem;
    }

    .prediction-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
    }

    .confidence-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.2rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }

    .confidence-high { background: #e8f5e9; color: #1b5e20; }
    .confidence-moderate { background: #fff8e1; color: #8a6a00; }
    .confidence-low { background: #ffebee; color: #b71c1c; }

    div[data-testid="stSidebar"] {
        background-color: #f5f5f7;
        border-right: 1px solid var(--border-color);
    }
    
    div[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .sidebar-card {
        background: transparent;
        border: none;
        box-shadow: none;
        padding: 1.2rem 0;
        border-bottom: 1px solid #d2d2d7;
        border-radius: 0;
        margin: 0;
    }

    .sidebar-card:last-child {
        border-bottom: none;
    }

    .sidebar-card h4 {
        margin: 0 0 0.6rem 0;
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 600;
    }

    .sidebar-card p {
        margin: 0;
        color: var(--text-secondary);
        line-height: 1.5;
        font-size: 0.9rem;
    }

    .stButton > button {
        background-color: var(--text-primary);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-weight: 500;
        font-size: 1.05rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #000000;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }

    .app-divider {
        height: 1px;
        background: var(--border-color);
        margin: 2.5rem 0;
    }

    .callout {
        background: #f5f5f7;
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 1.2rem;
        color: var(--text-primary);
        font-size: 1rem;
        margin-bottom: 2rem;
        box-shadow: none;
        text-align: center;
    }
    
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.2rem !important;
        }
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="hero-row">
        <div class="hero-badge">🌿 Dashboard</div>
        <h1>Green Consumer Behavior Predictor</h1>
        <p>An interactive dashboard to predict eco-friendly consumer preferences with clear insights, modern visuals, and beginner-friendly controls.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── DATA GENERATION & MODEL TRAINING (cached) ─────────────
@st.cache_resource
def load_model_and_data():
    """Generate dataset, train models, and return everything."""
    np.random.seed(42)
    N = 500

    scaled_models = ["Logistic Regression"]

    age               = np.random.randint(18, 70, N)
    income            = np.random.randint(20000, 150000, N)
    education_level   = np.random.choice([0, 1, 2, 3], N, p=[0.05, 0.30, 0.40, 0.25])
    environmental_concern = np.random.randint(1, 11, N)
    social_influence  = np.random.randint(1, 11, N)
    eco_awareness     = np.random.randint(1, 11, N)
    past_green_purchases = np.random.randint(0, 20, N)
    price_sensitivity = np.random.randint(1, 11, N)
    marketing_exposure = np.random.randint(1, 11, N)
    gender            = np.random.choice([0, 1], N)
    location          = np.random.choice([0, 1, 2], N, p=[0.20, 0.35, 0.45])

    score = (
        0.25 * (environmental_concern / 10) +
        0.20 * (eco_awareness / 10) +
        0.15 * (social_influence / 10) +
        0.15 * (past_green_purchases / 20) +
        0.10 * (income / 150000) +
        0.10 * (education_level / 3) +
        0.05 * (1 - price_sensitivity / 10) +
        np.random.normal(0, 0.05, N)
    )
    green_consumer = (score > 0.45).astype(int)

    df = pd.DataFrame({
        "Age": age, "Income": income, "Education_Level": education_level,
        "Environmental_Concern": environmental_concern,
        "Social_Influence": social_influence, "Eco_Awareness": eco_awareness,
        "Past_Green_Purchases": past_green_purchases,
        "Price_Sensitivity": price_sensitivity,
        "Marketing_Exposure": marketing_exposure,
        "Gender": gender, "Location": location,
        "Green_Consumer": green_consumer
    })

    X = df.drop("Green_Consumer", axis=1)
    y = df["Green_Consumer"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Decision Tree"      : DecisionTreeClassifier(max_depth=6, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    }

    trained = {}
    for name, m in models.items():
        if name in scaled_models:
            m.fit(X_train_sc, y_train)
            y_pred = m.predict(X_test_sc)
        else:
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        trained[name] = {"model": m, "acc": acc, "y_pred": y_pred}

    best_model_name = max(trained, key=lambda m: trained[m]["acc"])

    return df, X, y, X_train, X_test, y_train, y_test, scaler, trained, scaled_models, best_model_name


with st.spinner("Training models..."):
    df, X, y, X_train, X_test, y_train, y_test, scaler, trained, scaled_models, best_model_name = load_model_and_data()

# ─── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <h4>🌿 Green Consumer Predictor</h4>
        <p>Navigate through the app, compare models, and try live predictions with a clean dashboard-style interface.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Navigation")
    tab_choice = st.radio("Go to section:", [
        "🏠 Overview",
        "📊 EDA & Visualizations",
        "🤖 Model Comparison",
        "🔮 Live Prediction",
        "📄 Project Info"
    ])

    st.markdown("""
    <div class="sidebar-card">
    <h4>📌 Project Snapshot</h4>
    <p>
        Subject: Machine Learning<br>
        Topic: Green Consumer Behavior<br>
        Records: 500 consumers<br>
        Features: 11 input variables<br>
        Models: 4 classifiers
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>🏆 Best Model</h4>
        <p><strong>{best_model_name}</strong><br>Accuracy: <strong>{accuracy:.1f}%</strong></p>
    </div>
    """.format(best_model_name=best_model_name, accuracy=trained[best_model_name]['acc']*100), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════
if tab_choice == "🏠 Overview":
    st.markdown('<div class="section-title">📋 Problem Statement</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-card">
        <div class="subtle-text">
            <strong>Goal:</strong> Predict whether a consumer will prefer eco-friendly (green) products based on socio-demographic and behavioral factors using Machine Learning.<br><br>
            <strong>Target:</strong> 🟢 <strong>1 = Green Consumer</strong> — Prefers eco-friendly products &nbsp;&nbsp; 🔴 <strong>0 = Non-Green Consumer</strong> — Does not prefer eco-friendly products
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title section-spacing">🎯 Objectives</div>', unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    objectives = [
        ("📦", "Build a synthetic consumer dataset with realistic features"),
        ("🔍", "Perform EDA to understand patterns in green consumer behavior"),
        ("🤖", "Train and compare multiple ML classification models"),
        ("📈", "Evaluate models using Accuracy, F1-Score, and AUC"),
        ("🌟", "Identify top drivers of eco-friendly purchasing decisions"),
        ("🌐", "Deploy as an interactive web application using Streamlit"),
    ]
    for i, (icon, text) in enumerate(objectives):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="info-card green">
                <h4>{icon} {text}</h4>
                <p class="subtle-text">{text}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title section-spacing">💼 Real-World Importance</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""
        <div class="info-card">
            <h4>🏢 For Businesses</h4>
            <ul>
                <li>Target eco-conscious customers</li>
                <li>Develop green product lines</li>
                <li>Personalized marketing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card">
            <h4>🏛️ For Government</h4>
            <ul>
                <li>Design awareness campaigns</li>
                <li>Incentivize green choices</li>
                <li>Track sustainability progress</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="info-card">
            <h4>🌱 For NGOs</h4>
            <ul>
                <li>Identify at-risk communities</li>
                <li>Measure campaign effectiveness</li>
                <li>Allocate resources wisely</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title section-spacing">🗂️ Dataset Sample</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="section-title section-spacing">📊 Quick Metrics</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card"><div class="metric-kicker">Total Records</div><div class="metric-value">{len(df):,}</div><div class="metric-label">Synthetic consumers</div></div>
        <div class="metric-card"><div class="metric-kicker">Input Features</div><div class="metric-value">11</div><div class="metric-label">Behavior + demographic</div></div>
        <div class="metric-card"><div class="metric-kicker">Green Consumers</div><div class="metric-value">{df['Green_Consumer'].sum()}</div><div class="metric-label">Positive class count</div></div>
        <div class="metric-card"><div class="metric-kicker">Non-Green Consumers</div><div class="metric-value">{(df['Green_Consumer']==0).sum()}</div><div class="metric-label">Negative class count</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title section-spacing">📖 Features Explained</div>', unsafe_allow_html=True)
    feat_info = pd.DataFrame({
        "Feature": ["Age", "Income", "Education_Level", "Environmental_Concern",
                    "Social_Influence", "Eco_Awareness", "Past_Green_Purchases",
                    "Price_Sensitivity", "Marketing_Exposure", "Gender", "Location"],
        "Type": ["Numeric", "Numeric", "Ordinal (0–3)", "Ordinal (1–10)",
                 "Ordinal (1–10)", "Ordinal (1–10)", "Numeric (0–19)",
                 "Ordinal (1–10)", "Ordinal (1–10)", "Binary (0/1)", "Ordinal (0–2)"],
        "Description": [
            "Consumer age in years",
            "Annual income (₹)",
            "0=None, 1=High School, 2=Graduate, 3=Post-Graduate",
            "Concern for the environment (1=low, 10=high)",
            "Influence of peers/social media (1=low, 10=high)",
            "Knowledge about green products (1=low, 10=high)",
            "Number of eco-friendly products bought before",
            "Sensitivity to product price (1=very sensitive, 10=not sensitive)",
            "Exposure to green marketing campaigns",
            "0=Female, 1=Male",
            "0=Rural, 1=Suburban, 2=Urban"
        ]
    })
    st.table(feat_info)


# ═══════════════════════════════════════════════════════════
# TAB 2: EDA
# ═══════════════════════════════════════════════════════════
elif tab_choice == "📊 EDA & Visualizations":
    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-divider"></div>', unsafe_allow_html=True)

    viz = st.selectbox("Choose Visualization:", [
        "Target Distribution",
        "Feature Distributions",
        "Correlation Heatmap",
        "Boxplots by Consumer Type",
        "Income vs Environmental Concern (Scatter)",
    ])

    if viz == "Target Distribution":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        counts = df["Green_Consumer"].value_counts()
        axes[0].pie(counts, labels=["Green", "Non-Green"],
                    autopct="%1.1f%%", colors=["#4CAF50", "#FF7043"],
                    wedgeprops={"edgecolor": "white", "linewidth": 2})
        axes[0].set_title("Proportion")
        axes[1].bar(["Non-Green (0)", "Green (1)"], counts.values,
                    color=["#FF7043", "#4CAF50"], edgecolor="white")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Count")
        for i, v in enumerate(counts.values):
            axes[1].text(i, v + 3, str(v), ha="center", fontweight="bold")
        plt.suptitle("Target Variable: Green Consumer", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    elif viz == "Feature Distributions":
        feature = st.selectbox("Select Feature:", X.columns.tolist())
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[df["Green_Consumer"]==0][feature], bins=15, alpha=0.6,
                color="#FF7043", label="Non-Green")
        ax.hist(df[df["Green_Consumer"]==1][feature], bins=15, alpha=0.6,
                color="#4CAF50", label="Green")
        ax.set_title(f"Distribution of {feature.replace('_',' ')}", fontweight="bold")
        ax.set_xlabel(feature); ax.set_ylabel("Count"); ax.legend()
        st.pyplot(fig)

    elif viz == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(11, 8))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    mask=mask, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Correlation Heatmap", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    elif viz == "Boxplots by Consumer Type":
        feature = st.selectbox("Select Feature:", X.columns.tolist())
        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot(
            [df[df["Green_Consumer"]==0][feature],
             df[df["Green_Consumer"]==1][feature]],
            labels=["Non-Green", "Green"], patch_artist=True
        )
        bp["boxes"][0].set_facecolor("#FF7043")
        bp["boxes"][1].set_facecolor("#4CAF50")
        ax.set_title(f"{feature.replace('_',' ')} by Consumer Type", fontweight="bold")
        st.pyplot(fig)

    elif viz == "Income vs Environmental Concern (Scatter)":
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, color, marker in [(0, "#FF7043", "x"), (1, "#4CAF50", "o")]:
            sub = df[df["Green_Consumer"] == label]
            ax.scatter(sub["Income"], sub["Environmental_Concern"],
                       c=color, label="Green" if label else "Non-Green",
                       alpha=0.5, s=40, marker=marker)
        ax.set_xlabel("Income (₹)"); ax.set_ylabel("Environmental Concern (1–10)")
        ax.set_title("Income vs Environmental Concern", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.markdown('<div class="section-title section-spacing">📝 Basic Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 3: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════
elif tab_choice == "🤖 Model Comparison":
    st.markdown('<div class="section-title">🤖 Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-divider"></div>', unsafe_allow_html=True)

    model_names = list(trained.keys())
    accuracies  = [trained[m]["acc"] * 100 for m in model_names]

    c1, c2, c3, c4 = st.columns(4, gap="small")
    cols_list = [c1, c2, c3, c4]
    colors = ["#4CAF50", "#FFA726", "#42A5F5", "#EF5350"]
    for i, (name, col) in enumerate(zip(model_names, cols_list)):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-kicker">{name}</div>
            <div class="metric-value">{accuracies[i]:.1f}%</div>
            <div class="metric-label">Test accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(model_names, accuracies, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(60, 100)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Accuracy Comparison", fontweight="bold")
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", fontweight="bold", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

    # Feature importance
    st.markdown('<div class="section-title section-spacing">🌟 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    rf = trained["Random Forest"]["model"]
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feat_imp)))
    feat_imp.plot(kind="barh", ax=ax, color=colors_fi)
    ax.set_title("Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)

    # Confusion matrix for selected model
    st.markdown('<div class="section-title section-spacing">🧩 Confusion Matrix</div>', unsafe_allow_html=True)
    sel_model = st.selectbox("Select Model:", model_names)
    y_pred_sel = trained[sel_model]["y_pred"]
    cm = confusion_matrix(y_test, y_pred_sel)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Non-Green", "Green"],
                yticklabels=["Non-Green", "Green"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {sel_model}", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    # ROC Curve
    st.markdown('<div class="section-title section-spacing">📈 ROC Curve</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0,1],[0,1],"k--", alpha=0.4, label="Random (AUC=0.50)")
    for name, color in zip(model_names, colors):
        m = trained[name]["model"]
        try:
            if name in scaled_models:
                y_prob = m.predict_proba(scaler.transform(X_test))[:,1]
            else:
                y_prob = m.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
        except Exception:
            pass
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


# ═══════════════════════════════════════════════════════════
# TAB 4: LIVE PREDICTION
# ═══════════════════════════════════════════════════════════
elif tab_choice == "🔮 Live Prediction":
    st.markdown('<div class="section-title">🔮 Predict Green Consumer Behavior</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout">Fill in the consumer details and click <strong>Predict</strong> to get the result.</div>', unsafe_allow_html=True)

    selected_model = st.selectbox("Choose Model for Prediction:", list(trained.keys()))

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            st.markdown("**👤 Demographics**")
            age_in       = st.slider("Age", 18, 70, 28)
            income_in    = st.number_input("Annual Income (₹)", 20000, 150000, 75000, step=5000)
            gender_in    = st.selectbox("Gender", ["Female (0)", "Male (1)"])
            location_in  = st.selectbox("Location", ["Rural (0)", "Suburban (1)", "Urban (2)"])
            edu_in       = st.selectbox("Education Level",
                                        ["No Formal (0)", "High School (1)",
                                         "Graduate (2)", "Post-Graduate (3)"])

        with c2:
            st.markdown("**🌿 Green Behavior Scores**")
            env_concern  = st.slider("Environmental Concern (1–10)", 1, 10, 7)
            eco_aware    = st.slider("Eco Awareness (1–10)", 1, 10, 7)
            past_buy     = st.slider("Past Green Purchases (0–19)", 0, 19, 10)
            soc_inf      = st.slider("Social Influence (1–10)", 1, 10, 6)

        with c3:
            st.markdown("**💰 Other Factors**")
            price_sens   = st.slider("Price Sensitivity (1–10)", 1, 10, 4)
            mkt_exp      = st.slider("Marketing Exposure (1–10)", 1, 10, 6)

        submitted = st.form_submit_button("🔍 Predict Now", use_container_width=True)

    if submitted:
        gender_val   = int(gender_in.split("(")[1].replace(")", ""))
        location_val = int(location_in.split("(")[1].replace(")", ""))
        edu_val      = int(edu_in.split("(")[1].replace(")", ""))

        user_data = pd.DataFrame([{
            "Age": age_in, "Income": income_in, "Education_Level": edu_val,
            "Environmental_Concern": env_concern, "Social_Influence": soc_inf,
            "Eco_Awareness": eco_aware, "Past_Green_Purchases": past_buy,
            "Price_Sensitivity": price_sens, "Marketing_Exposure": mkt_exp,
            "Gender": gender_val, "Location": location_val
        }])[X.columns]

        model = trained[selected_model]["model"]
        if selected_model in scaled_models:
            user_sc = scaler.transform(user_data)
            pred = model.predict(user_sc)[0]
            prob = model.predict_proba(user_sc)[0][1] * 100
        else:
            pred = model.predict(user_data)[0]
            prob = model.predict_proba(user_data)[0][1] * 100

        if prob > 80:
            confidence_label = "High confidence"
            confidence_message = "The model is strongly confident in this prediction."
            confidence_color = "success"
        elif 60 <= prob <= 80:
            confidence_label = "Moderate confidence"
            confidence_message = "The model has a reasonable level of confidence here."
            confidence_color = "warning"
        else:
            confidence_label = "Low confidence"
            confidence_message = "The model is less certain, so treat this prediction cautiously."
            confidence_color = "error"

        st.markdown('<div class="app-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

        result_col, detail_col = st.columns([1.7, 1], gap="large")
        with result_col:
            if pred == 1:
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-banner-green">
                        <div class="prediction-title">🌿 Green Consumer</div>
                        <div class="prediction-subtitle">This person is likely to prefer eco-friendly products.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-banner-red">
                        <div class="prediction-title">❌ Non-Green Consumer</div>
                        <div class="prediction-subtitle">This person is unlikely to prefer eco-friendly products.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with detail_col:
            if confidence_color == "success":
                confidence_class = "confidence-high"
                confidence_icon = "🟢"
            elif confidence_color == "warning":
                confidence_class = "confidence-moderate"
                confidence_icon = "🟡"
            else:
                confidence_class = "confidence-low"
                confidence_icon = "🔴"

            st.markdown(f"""
            <div class="prediction-card">
                <div class="confidence-chip {confidence_class}">{confidence_icon} {confidence_label}</div>
                <div class="metric-grid" style="grid-template-columns: 1fr; gap: 0.7rem;">
                    <div class="metric-card"><div class="metric-kicker">Green Probability</div><div class="metric-value">{prob:.1f}%</div><div class="metric-label">Chance of being green</div></div>
                    <div class="metric-card"><div class="metric-kicker">Non-Green Probability</div><div class="metric-value">{100-prob:.1f}%</div><div class="metric-label">Chance of being non-green</div></div>
                    <div class="metric-card"><div class="metric-kicker">Model Used</div><div class="metric-value">{selected_model}</div><div class="metric-label">Prediction model</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if confidence_color == "success":
            st.success(f"{confidence_label}: {confidence_message}")
        elif confidence_color == "warning":
            st.warning(f"{confidence_label}: {confidence_message}")
        else:
            st.error(f"{confidence_label}: {confidence_message}")

        # Probability gauge
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh([""], [prob], color="#4CAF50", height=0.4)
        ax.barh([""], [100 - prob], left=[prob], color="#FF7043", height=0.4)
        ax.set_xlim(0, 100); ax.set_xlabel("Probability (%)")
        ax.set_title("Green vs Non-Green Probability", fontweight="bold")
        ax.axvline(50, color="black", linewidth=1, linestyle="--")
        ax.text(prob/2, 0, f"{prob:.0f}%", ha="center", va="center",
                fontweight="bold", color="white", fontsize=12)
        ax.text(prob + (100-prob)/2, 0, f"{100-prob:.0f}%",
                ha="center", va="center", fontweight="bold", color="white", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        # Personalized tips
        st.markdown('<div class="section-title section-spacing">💡 Personalized Insights</div>', unsafe_allow_html=True)
        if env_concern < 5:
            st.warning("🌍 Low environmental concern — Education campaigns could help.")
        if eco_aware < 5:
            st.warning("📚 Low eco awareness — Targeted green product information recommended.")
        if past_buy < 3:
            st.info("🛒 Very few past green purchases — Incentives or discounts could convert this consumer.")
        if pred == 1:
            st.success("✅ This consumer is a great target for eco-friendly product marketing!")
        else:
            st.error("⚠️ This consumer may need more awareness and incentives before buying green.")


# ═══════════════════════════════════════════════════════════
# TAB 5: PROJECT INFO
# ═══════════════════════════════════════════════════════════
elif tab_choice == "📄 Project Info":
    st.markdown('<div class="section-title">📄 Project Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-divider"></div>', unsafe_allow_html=True)
    st.code(f"""
  [Data Generation] → [EDA] → [Preprocessing] → [Model Training]
                                                        │
                             ┌──────────────────────────┤
                             │  Logistic Regression      │
                             │  Decision Tree            │
                             │  {best_model_name} ← Best
                             │  Gradient Boosting        │
                             └──────────────────────────┘
                                          │
                                    [Evaluation]
                              Accuracy / AUC / F1-Score
                                          │
                                  [Prediction API]
                                          │
                              [Streamlit Web App (this!)]
    """, language="text")

    st.markdown('<div class="section-title section-spacing">📌 Conclusion</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-card green">
        <h4>Key Takeaways</h4>
        <ul>
            <li>Environmental Concern, Eco Awareness, and Past Green Purchases are the <strong>top predictors</strong>.</li>
            <li><strong>{best_model_name}</strong> achieves the <strong>highest accuracy</strong>.</li>
            <li>The model can reliably classify green vs non-green consumers.</li>
            <li>Actionable insights help businesses and governments design targeted strategies.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title section-spacing">🚀 Future Scope</div>', unsafe_allow_html=True)
    future = [
        "Use **real survey or e-commerce purchase data** for better generalization",
        "Apply **Deep Learning (ANN/LSTM)** for higher accuracy",
        "Add **SHAP explainability** for individual prediction transparency",
        "Extend to **multi-class prediction** (highly green, moderately green, not green)",
        "Build a **REST API** using Flask or FastAPI for integration",
        "Add **time-series tracking** to monitor behavior change over time",
    ]
    for item in future:
        st.markdown(f"✅ {item}")

    st.markdown('<div class="section-title section-spacing">📚 Libraries Used</div>', unsafe_allow_html=True)
    libs = {
        "pandas": "Data manipulation and analysis",
        "numpy": "Numerical computing",
        "scikit-learn": "ML model training and evaluation",
        "matplotlib": "Static visualizations",
        "seaborn": "Statistical visualizations",
        "streamlit": "Interactive web application",
    }
    lib_df = pd.DataFrame({"Library": libs.keys(), "Purpose": libs.values()})
    st.table(lib_df)
