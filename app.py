
import streamlit as st
import joblib
import numpy as np
import nltk
from pythainlp.tokenize import word_tokenize

nltk.download('punkt')

# โหลดโมเดลและเวกเตอร์
vectorizer_tf = joblib.load("vectorizer_tf.pkl")
naive_model_tf_smote = joblib.load("naive_model_tf_smote.pkl")
svm_model_smote_embedding = joblib.load("svm_model_smote_embedding.pkl")
rf_model_rus_embedding = joblib.load("rf_model_rus_embedding.pkl")

# โหลด document embedding function
def document_embedding(tokens):
    # โปรดแก้ให้ตรงกับที่คุณใช้จริง
    return np.mean([np.random.rand(300) for _ in tokens], axis=0)  # จำลอง 300-dim vector

# โหลดน้ำหนักจาก cross-validation
weight_tf = 0.31  # ตัวอย่าง
weight_svm = 0.35
weight_rf = 0.34

# ---------- pipeline ----------
def predict_pipeline_tf(headline):
    X = vectorizer_tf.transform([headline])
    return naive_model_tf_smote.predict(X)[0]

def predict_pipeline_svm(headline):
    tokens = word_tokenize(headline, engine="newmm")
    X = document_embedding(tokens).reshape(1, -1)
    return svm_model_smote_embedding.predict(X)[0]

def predict_pipeline_rf(headline):
    tokens = word_tokenize(headline, engine="newmm")
    X = document_embedding(tokens).reshape(1, -1)
    return rf_model_rus_embedding.predict(X)[0]

# ---------- ensemble ----------
def ensemble_predict(headline):
    pred_tf = predict_pipeline_tf(headline)
    pred_svm = predict_pipeline_svm(headline)
    pred_rf = predict_pipeline_rf(headline)

    vote = {}
    vote[pred_tf] = vote.get(pred_tf, 0) + weight_tf
    vote[pred_svm] = vote.get(pred_svm, 0) + weight_svm
    vote[pred_rf] = vote.get(pred_rf, 0) + weight_rf

    final = max(vote, key=vote.get)

    details = f"Naïve Bayes: {pred_tf} (w={weight_tf:.2f})\n"
    details += f"SVM: {pred_svm} (w={weight_svm:.2f})\n"
    details += f"Random Forest: {pred_rf} (w={weight_rf:.2f})\n"
    details += f"\n🎯 Final Prediction: {final}"
    return details

# ---------- Streamlit UI ----------
st.title("📰 Thai News Headline Classifier (Weighted Voting)")
st.markdown("This app classifies Thai news headlines into categories using 3 ML models.")

mode = st.radio("Mode", ["Single Model", "Ensemble Voting"])
selected_model = None

if mode == "Single Model":
    selected_model = st.selectbox("Select Model", ["Naïve Bayes", "SVM", "Random Forest"])

headline = st.text_input("Enter Thai news headline:")

if st.button("Predict"):
    if mode == "Single Model":
        if selected_model == "Naïve Bayes":
            result = predict_pipeline_tf(headline)
        elif selected_model == "SVM":
            result = predict_pipeline_svm(headline)
        elif selected_model == "Random Forest":
            result = predict_pipeline_rf(headline)
        st.success(f"Predicted: {result}")
    else:
        result = ensemble_predict(headline)
        st.success(result)
