# 🌿 Green Consumer Behavior Prediction
### A Complete Machine Learning Project | College Submission

---

## 📌 Problem Statement
Predict whether a consumer will prefer **eco-friendly (green) products** based on
socio-demographic and behavioral factors using supervised Machine Learning.

**Target Variable:**
- `1` → Green Consumer (prefers eco-friendly products)
- `0` → Non-Green Consumer (does not prefer eco-friendly products)

---

## 🎯 Objectives
1. Build a synthetic consumer dataset with realistic features
2. Perform EDA to identify patterns in green consumer behavior
3. Train and compare multiple ML classification models
4. Evaluate using Accuracy, F1-Score, and ROC-AUC
5. Identify the top drivers of eco-friendly purchasing decisions
6. Deploy as an interactive web app using Streamlit

---

## 📂 Project Structure
```
green_consumer_project/
│
├── green_consumer_prediction.py   ← Main ML project (run this first)
├── streamlit_app.py               ← Interactive web application
├── README.md                      ← This file
│
└── output_graphs/                 ← Auto-generated after running main script
    ├── green_consumer_dataset.csv
    ├── 01_target_distribution.png
    ├── 02_feature_distributions.png
    ├── 03_correlation_heatmap.png
    ├── 04_boxplots.png
    ├── 05_model_comparison.png
    ├── 06_confusion_matrices.png
    ├── 07_roc_curves.png
    ├── 08_feature_importance.png
    └── 09_learning_curve.png
```

---

## 🗂️ Dataset Details

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Consumer age (18–70) |
| Income | Numeric | Annual income in ₹ |
| Education_Level | Ordinal (0–3) | 0=None, 1=High School, 2=Graduate, 3=PG |
| Environmental_Concern | Score (1–10) | Concern for the environment |
| Social_Influence | Score (1–10) | Peer/social media influence |
| Eco_Awareness | Score (1–10) | Knowledge of green products |
| Past_Green_Purchases | Numeric (0–19) | Number of previous eco-purchases |
| Price_Sensitivity | Score (1–10) | Sensitivity to product price |
| Marketing_Exposure | Score (1–10) | Exposure to green campaigns |
| Gender | Binary (0/1) | 0=Female, 1=Male |
| Location | Ordinal (0–2) | 0=Rural, 1=Suburban, 2=Urban |
| **Green_Consumer** | **Binary (0/1)** | **TARGET VARIABLE** |

**Records:** 500 consumers | **Features:** 11 inputs + 1 target

---

## 🤖 ML Models Used

| Model | Test Accuracy | AUC |
|---|---|---|
| Logistic Regression | ~94% | ~0.97 |
| Support Vector Machine | ~93% | ~0.94 |
| Random Forest | ~86% | ~0.93 |
| Gradient Boosting | ~85% | ~0.92 |
| K-Nearest Neighbors | ~86% | ~0.87 |
| Decision Tree | ~77% | ~0.73 |

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn streamlit
```

### 2. Run the Main ML Project
```bash
python green_consumer_prediction.py
```
→ Generates all graphs and prints results in the terminal.

### 3. Launch the Streamlit Web App
```bash
streamlit run streamlit_app.py
```
→ Opens an interactive browser app for EDA, model comparison, and live prediction.

---

## 📊 Key Results
- **Best Model:** Logistic Regression (94% accuracy, AUC = 0.968)
- **Top Features:** Environmental Concern, Eco Awareness, Past Green Purchases
- **Evaluation:** 5-Fold Cross-Validation confirms generalizability

---

## 🔮 Real-World Use Cases
- **Retail Businesses** → Target eco-conscious customers
- **Government Agencies** → Design green awareness campaigns
- **NGOs** → Identify communities that need sustainability education
- **Marketing Teams** → Personalized green product recommendations

---

## 🚀 Future Scope
1. Integrate real survey / e-commerce purchase data
2. Apply Deep Learning (ANN) for higher accuracy
3. Add SHAP values for prediction explainability
4. Extend to multi-class prediction (high/medium/low green)
5. Deploy as REST API using Flask or FastAPI
6. Build a mobile app version

---

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Data:** pandas, numpy
- **ML:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Web App:** Streamlit

---

*Project by Charan Rakindi | Computer Science Engineering*
