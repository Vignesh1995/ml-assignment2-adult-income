import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

TARGET = "income"
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
LABELS = ["<=50K", ">50K"]

st.set_page_config(page_title="Adult Income Classifier", layout="wide")
st.title("💰 Adult Income Prediction – Model Evaluation")
st.caption("Predict whether annual income exceeds $50K (UCI Adult dataset). "
           "Pre-trained models are loaded from disk; nothing is retrained.")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y):
    pred = model.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, scores(model, X)),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return pred, metrics


with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    choice = st.selectbox("Select model", list(MODEL_FILES))
    compare_all = st.checkbox("Compare all models", value=True)
    st.markdown("CSV must contain the 14 feature columns and the "
                f"`{TARGET}` column (0 = <=50K, 1 = >50K). "
                "Use `test_data.csv` from the repo.")

if uploaded is None:
    st.info("⬅️ Upload a test CSV in the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"Target column `{TARGET}` not found in the uploaded file.")
    st.stop()

if df[TARGET].dtype == object:
    df[TARGET] = df[TARGET].str.strip().str.rstrip(".").eq(">50K").astype(int)
X, y = df.drop(columns=TARGET), df[TARGET]

st.subheader("Data preview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Features", X.shape[1])
c3.metric("% >50K", f"{y.mean() * 100:.1f}%")
st.dataframe(df.head(10), width='stretch')

model = load_model(MODEL_FILES[choice])
try:
    pred, metrics = evaluate(model, X, y)
except Exception as exc:
    st.error(f"Could not score the data with this model: {exc}")
    st.stop()

st.subheader(f"Results – {choice}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=LABELS, cmap="Blues", ax=ax)
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, pred, target_names=LABELS,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), width='stretch')

if compare_all:
    st.subheader("All models on the uploaded data")
    rows = []
    for name, path in MODEL_FILES.items():
        _, m = evaluate(load_model(path), X, y)
        rows.append({"ML Model Name": name, **m})
    st.dataframe(pd.DataFrame(rows).set_index("ML Model Name").round(4),
                 width='stretch')
