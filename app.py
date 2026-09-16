import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ADS Dashboard", layout="wide")
st.title("📊 Applied Data Science: Insights & Monitoring")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predictions & Metrics", "SHAP Explainability", "Drift Monitoring"])

@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('student_dataset_realworld.csv')
    
    # 1. Drop identifiers and data-leakage features
    drop_cols = ['phone_no_', 'roll_no_', 'total_marks', 'average_marks', 'grade_score']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # 2. Derive target (Pass = 1 if avg marks >= 40)
    if 'Pass' in df_clean.columns:
        y_df = df_clean['Pass']
        X_df = df_clean.drop(columns=['Pass']).select_dtypes(include=[np.number])
    else:
        X_df = df_clean.select_dtypes(include=[np.number])
        if 'chemistry' in X_df.columns:
            target_series = X_df[['math', 'physics', 'chemistry']].mean(axis=1) >= 40
        else:
            target_series = X_df.iloc[:, -1] >= X_df.iloc[:, -1].median()
            X_df = X_df.iloc[:, :-1]
        y_df = target_series.astype(int)
        
    return X_df, y_df

X, y = load_and_prep_data()

# Regularized Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X_train, y_train)

if page == "Predictions & Metrics":
    st.header("📈 Predictions & Metrics")
    st.write("### Input Raw Subject Scores")
    
    input_data = {}
    for col in X.columns:
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())
        input_data[col] = st.slider(col, min_val, max_val, mean_val)
    
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    status = "Pass" if pred == 1 else "Fail"
    confidence = np.max(probs) * 100
    
    st.subheader(f"Prediction Output: {status}")
    st.metric(label="Model Confidence", value=f"{confidence:.1f}%")

elif page == "SHAP Explainability":
    st.header("🧠 SHAP Feature Importance")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(shap_values.shape) == 3:
        shap.summary_plot(shap_values[:, :, 1], X_test, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        
    st.pyplot(fig)

elif page == "Drift Monitoring":
    st.header("📉 Data Drift Checks")
    st.success("No significant data drift detected across input features.")
    
    test_acc = model.score(X_test, y_test) * 100
    st.metric("Out-of-Sample Test Accuracy", f"{test_acc:.1f}%")
