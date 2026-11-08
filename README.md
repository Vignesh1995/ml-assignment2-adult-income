# Adult Income Classification – ML Assignment 2 (M.Tech AIML, BITS WILP)

## a. Problem statement
Predict whether a person's annual income exceeds **$50K** from census attributes
(binary classification). Five classifiers are trained on the same dataset,
compared on six metrics, and served through a Streamlit app that evaluates the
saved models on an uploaded test CSV.

## b. Dataset description
- **Source:** UCI Machine Learning Repository – Adult (Census Income) dataset.
- **Instances:** 48,842 (32,561 train / 16,281 test, using the official split).
- **Features:** 14 (6 numeric, 8 categorical) – age, workclass, fnlwgt, education,
  education_num, marital_status, occupation, relationship, race, sex,
  capital_gain, capital_loss, hours_per_week, native_country.
- **Target:** `income` – 1 if >50K, else 0 (≈24% positive, so moderately imbalanced).
- **Preprocessing:** missing values (`?`) → `"Unknown"`; numeric features
  standardised; categorical features one-hot encoded. The preprocessing is saved
  inside each model pipeline, so the app accepts raw feature columns.
- `test_data.csv` is the official 16,281-row test split.

## c. GitHub repository link
https://github.com/Vignesh1995/ml-assignment2-adult-income

Live app: https://ml-assignment2-adult-income-qyprfmgbq4qkgmo79aiwib.streamlit.app/

```
├── app.py              # Streamlit app
├── train.py            # training + evaluation script
├── requirements.txt
├── README.md
├── test_data.csv       # held-out test data
├── metrics.csv         # metrics table
└── model/              # saved pipelines (.joblib) for all 5 classifiers
```

Run locally: `pip install -r requirements.txt && python train.py && streamlit run app.py`

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8530 | 0.9055 | 0.7306 | 0.5985 | 0.6580 | 0.5701 |
| Decision Tree | 0.8566 | 0.9000 | 0.7914 | 0.5335 | 0.6374 | 0.5688 |
| KNN | 0.8442 | 0.8892 | 0.6942 | 0.6084 | 0.6485 | 0.5509 |
| Naive Bayes | 0.5532 | 0.7639 | 0.3389 | 0.9376 | 0.4979 | 0.3320 |
| Support Vector Machine | 0.8597 | 0.9006 | 0.7611 | 0.5915 | 0.6657 | 0.5860 |

Hyperparameters: Logistic Regression (max_iter=1000), Decision Tree (max_depth=8),
KNN (k=15), Gaussian Naive Bayes (defaults), SVM (RBF kernel, C=1).
AUC for SVM uses the decision-function score.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, stable baseline with the best AUC (0.9055); the linear boundary suits the one-hot features well. Recall (0.60) is modest because of class imbalance. |
| Decision Tree | High precision (0.79) but lowest recall (0.53) among the good models, so it is conservative about predicting >50K. Depth limit of 8 prevents overfitting; interpretable. |
| KNN | Slightly weaker than the others (accuracy 0.844, MCC 0.551). Distance in a high-dimensional one-hot space is less informative; it is also slow at prediction time. |
| Naive Bayes | Clearly worst (accuracy 0.553, MCC 0.332). Gaussian assumption is violated by one-hot binary features, and independence is unrealistic; it over-predicts >50K (recall 0.94, precision 0.34). |
| Support Vector Machine | Best accuracy (0.8597), F1 (0.6657) and MCC (0.5860); the RBF kernel captures non-linear interactions. Costlier to train than the others. |
| **Overall winner** | **Support Vector Machine** – best Accuracy, F1 and MCC, with AUC within 0.005 of the best (Logistic Regression). Logistic Regression is the best lightweight alternative. |

## Streamlit app features
- CSV upload of test data (must include the target column `income`)
- Model selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC display
- Confusion matrix and classification report
- Optional comparison table across all five models
