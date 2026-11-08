"""Train five classifiers on the UCI Adult Income dataset and save them."""
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

COLS = ["age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"]
TARGET = "income"


def load(path, skip=0):
    df = pd.read_csv(path, names=COLS, skiprows=skip, skipinitialspace=True,
                     na_values="?")
    df[TARGET] = df[TARGET].str.strip().str.rstrip(".")
    df[TARGET] = (df[TARGET] == ">50K").astype(int)
    return df


def make_preprocessor(X):
    cat = X.select_dtypes(include="object").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def main():
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    train = load(base + "adult.data")
    test = load(base + "adult.test", skip=1)
    # Fill missing categoricals with a label so rows are kept
    for df in (train, test):
        df.fillna("Unknown", inplace=True)
    test.to_csv("test_data.csv", index=False)

    X_tr, y_tr = train.drop(columns=TARGET), train[TARGET]
    X_te, y_te = test.drop(columns=TARGET), test[TARGET]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, random_state=42),
    }

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("prep", make_preprocessor(X_tr)), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        score = (pipe.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba")
                 and name != "Support Vector Machine"
                 else pipe.decision_function(X_te))
        rows.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_te, pred),
            "AUC": roc_auc_score(y_te, score),
            "Precision": precision_score(y_te, pred),
            "Recall": recall_score(y_te, pred),
            "F1": f1_score(y_te, pred),
            "MCC": matthews_corrcoef(y_te, pred),
        })
        joblib.dump(pipe, f"model/{name.lower().replace(' ', '_')}.joblib",
                    compress=3)
        print(rows[-1])

    res = pd.DataFrame(rows).round(4)
    res.to_csv("metrics.csv", index=False)
    print(res.to_markdown(index=False))


if __name__ == "__main__":
    main()
