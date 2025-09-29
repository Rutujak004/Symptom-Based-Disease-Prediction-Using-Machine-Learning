import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------
# Load Data Function
# -----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Training.csv")
    df = df.dropna(axis=1)

    # Features and Target
    X = df.iloc[:, :-1]
    y = df["prognosis"].str.strip()  # remove trailing spaces

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Save encoder in session state for decoding
    st.session_state['label_encoder'] = le

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_test, y_test, X.columns

# -----------------------------------------------------------
# Train Models
# -----------------------------------------------------------
@st.cache_resource
def train_models(X_train, y_train):
    dt = DecisionTreeClassifier().fit(X_train, y_train)
    rf = RandomForestClassifier().fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    return dt, rf, nb

# -----------------------------------------------------------
# Predict Disease
# -----------------------------------------------------------
def predict_disease(model, symptoms, all_symptoms):
    input_data = [0] * len(all_symptoms)
    for symptom in symptoms:
        if symptom in all_symptoms:
            idx = list(all_symptoms).index(symptom)
            input_data[idx] = 1

    prediction = model.predict([input_data])
    disease = st.session_state['label_encoder'].inverse_transform(prediction)[0]
    return disease

# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
def main():
    st.title("🩺 Symptom-Based Disease Prediction")

    # Load Data
    X_train, y_train, X_test, y_test, all_symptoms = load_data()

    # Train Models
    dt, rf, nb = train_models(X_train, y_train)

    # Accuracy
    st.subheader("📊 Model Accuracy")
    st.write(f"Decision Tree: {accuracy_score(y_test, dt.predict(X_test)):.2f}")
    st.write(f"Random Forest: {accuracy_score(y_test, rf.predict(X_test)):.2f}")
    st.write(f"Naive Bayes: {accuracy_score(y_test, nb.predict(X_test)):.2f}")

    # Symptom Selection
    st.subheader("📝 Enter Symptoms")
    selected_symptoms = st.multiselect(
        "Select the symptoms you are experiencing:",
        options=all_symptoms
    )

    if st.button("🔍 Predict Disease"):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            st.success("✅ Predictions:")
            st.write(f"**Decision Tree:** {predict_disease(dt, selected_symptoms, all_symptoms)}")
            st.write(f"**Random Forest:** {predict_disease(rf, selected_symptoms, all_symptoms)}")
            st.write(f"**Naive Bayes:** {predict_disease(nb, selected_symptoms, all_symptoms)}")

if __name__ == "__main__":
    main()
