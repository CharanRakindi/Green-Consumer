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
    .main-header {
        background: linear-gradient(135deg, #1B5E20, #4CAF50);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #F1F8E9;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .predict-box-green {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #1B5E20;
    }
    .predict-box-red {
        background: linear-gradient(135deg, #FBE9E7, #FFCCBC);
        border: 2px solid #FF7043;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #BF360C;
    }
    .sidebar-info {
        background: #E8F5E9;
        padding: 12px;
        border-radius: 8px;
        font-size: 13px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:32px">🌿 Green Consumer Behavior Predictor</h1>
    <p style="margin:5px 0 0 0; opacity:0.9; font-size:15px">
        ML-powered tool to predict eco-friendly consumer preferences
    </p>
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
    st.header("⚙️ Navigation")
    tab_choice = st.radio("Go to section:", [
        "🏠 Overview",
        "📊 EDA & Visualizations",
        "🤖 Model Comparison",
        "🔮 Live Prediction",
        "📄 Project Info"
    ])

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
    <b>📌 Project Info</b><br>
    Subject: Machine Learning<br>
    Topic: Green Consumer Behavior<br>
    Records: 500 consumers<br>
    Features: 11 input variables<br>
    Models: 4 classifiers
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.success(f"🏆 Best Model: **{best_model_name}**\n\nAccuracy: **{trained[best_model_name]['acc']*100:.1f}%**")


# ═══════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════
if tab_choice == "🏠 Overview":
    st.subheader("📋 Problem Statement")
    st.info("""
    **Goal:** Predict whether a consumer will prefer eco-friendly (green) products
    based on socio-demographic and behavioral factors using Machine Learning.

    **Target:**  
    🟢 **1 = Green Consumer** — Prefers eco-friendly products  
    🔴 **0 = Non-Green Consumer** — Does not prefer eco-friendly products
    """)

    st.subheader("🎯 Objectives")
    cols = st.columns(2)
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
            st.markdown(f"**{icon} {text}**")

    st.subheader("💼 Real-World Importance")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **🏢 For Businesses**
        - Target eco-conscious customers
        - Develop green product lines
        - Personalized marketing
        """)
    with c2:
        st.markdown("""
        **🏛️ For Government**
        - Design awareness campaigns
        - Incentivize green choices
        - Track sustainability progress
        """)
    with c3:
        st.markdown("""
        **🌱 For NGOs**
        - Identify at-risk communities
        - Measure campaign effectiveness
        - Allocate resources wisely
        """)

    st.subheader("🗂️ Dataset Sample")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Input Features", "11")
    c3.metric("Green Consumers", f"{df['Green_Consumer'].sum()}")
    c4.metric("Non-Green Consumers", f"{(df['Green_Consumer']==0).sum()}")

    st.subheader("📖 Features Explained")
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
    st.subheader("📊 Exploratory Data Analysis")

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

    st.subheader("📝 Basic Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 3: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════
elif tab_choice == "🤖 Model Comparison":
    st.subheader("🤖 Model Performance Comparison")

    model_names = list(trained.keys())
    accuracies  = [trained[m]["acc"] * 100 for m in model_names]

    c1, c2, c3, c4 = st.columns(4)
    cols_list = [c1, c2, c3, c4]
    colors = ["#4CAF50", "#FFA726", "#42A5F5", "#EF5350"]
    for i, (name, col) in enumerate(zip(model_names, cols_list)):
        col.metric(name.split()[0], f"{accuracies[i]:.1f}%")

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
    st.subheader("🌟 Feature Importance (Random Forest)")
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
    st.subheader("🧩 Confusion Matrix")
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
    st.subheader("📈 ROC Curve")
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
    st.subheader("🔮 Predict Green Consumer Behavior")
    st.write("Fill in the consumer details and click **Predict** to get the result.")

    selected_model = st.selectbox("Choose Model for Prediction:", list(trained.keys()))

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

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

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        c1, c2 = st.columns([2, 1])
        with c1:
            if pred == 1:
                st.markdown(f"""
                <div class="predict-box-green">
                    🌿 GREEN CONSUMER<br>
                    <small>This person is likely to prefer eco-friendly products</small>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="predict-box-red">
                    ❌ NON-GREEN CONSUMER<br>
                    <small>This person is unlikely to prefer eco-friendly products</small>
                </div>""", unsafe_allow_html=True)

        with c2:
            st.metric("Green Probability", f"{prob:.1f}%")
            st.metric("Non-Green Probability", f"{100-prob:.1f}%")
            st.metric("Model Used", selected_model.split()[0])

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
        st.subheader("💡 Personalized Insights")
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
    st.subheader("📄 Project Architecture")
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

    st.subheader("📌 Conclusion")
    st.success("""
    - Environmental Concern, Eco Awareness, and Past Green Purchases are the **top predictors**.
    - {best_model_name} achieves the **highest accuracy**.
    - The model can reliably classify green vs non-green consumers.
    - Actionable insights help businesses and governments design targeted strategies.
    """.format(best_model_name=best_model_name))

    st.subheader("🚀 Future Scope")
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

    st.subheader("📚 Libraries Used")
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
