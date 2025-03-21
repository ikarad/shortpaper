
import streamlit as st
import joblib
import nltk
import numpy as np
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# โหลดโมเดลที่บันทึกไว้
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
nb_model = joblib.load("nb_model.pkl")
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# ฟังก์ชันเตรียมข้อความก่อนนำไปใช้กับโมเดล
def preprocess_text(text):
    tokens = word_tokenize(text)
    return " ".join(tokens)

# ฟังก์ชันทำนายผลจากโมเดลแต่ละตัว
def predict_pipeline_tf(headline):
    headline = preprocess_text(headline)
    features = tfidf_vectorizer.transform([headline])
    return nb_model.predict(features)[0]

def predict_pipeline_svm(headline):
    headline = preprocess_text(headline)
    features = tfidf_vectorizer.transform([headline])
    return svm_model.predict(features)[0]

def predict_pipeline_rf(headline):
    headline = preprocess_text(headline)
    features = tfidf_vectorizer.transform([headline])
    return rf_model.predict(features)[0]

# ฟังก์ชันการทำนายในโหมด Single Model และ Ensemble
def predict_category(headline, model_name):
    if model_name == "Naïve Bayes":
        return predict_pipeline_tf(headline)
    elif model_name == "SVM":
        return predict_pipeline_svm(headline)
    elif model_name == "Random Forest":
        return predict_pipeline_rf(headline)

# ฟังก์ชัน Ensemble Voting (Majority Vote)
def ensemble_predict(headline):
    predictions = [
        predict_pipeline_tf(headline),
        predict_pipeline_svm(headline),
        predict_pipeline_rf(headline)
    ]
    return max(set(predictions), key=predictions.count)

# ===========================
# 🔹 Streamlit UI
# ===========================

st.title("📰 Thai News Headline Classifier 🚀")
st.markdown("Classify Thai news headlines into categories.")

mode = st.radio("Choose mode:", ["Single Model", "Ensemble Voting"])
selected_model = None

if mode == "Single Model":
    selected_model = st.selectbox("Select a model:", ["Naïve Bayes", "SVM", "Random Forest"])

headline = st.text_input("Enter a news headline:")

if st.button("Predict"):
    if mode == "Single Model" and selected_model:
        result = predict_category(headline, selected_model)
        st.success(f"Predicted Category: {result}")
    elif mode == "Ensemble Voting":
        result = ensemble_predict(headline)
        st.success(f"Predicted Category (Ensemble): {result}")
