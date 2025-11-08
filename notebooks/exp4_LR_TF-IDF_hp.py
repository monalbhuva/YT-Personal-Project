import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Set MLflow Tracking URI & DAGsHub integration
MLFLOW_TRACKING_URI = "https://dagshub.com/monalAJ/YT-Personal-Project.mlflow"
dagshub.init(repo_owner="monalAJ", repo_name="YT-Personal-Project", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("LR BoW Hyperparameter Tuning")


# ==========================
# Text Preprocessing
# ==========================
def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except:
        return comment


# Balancing dataset
from sklearn.utils import resample
def balance_dataset(df, target_col='category'):
    classes = df[target_col].unique()
    target_size = int(df[target_col].value_counts().mean())

    balanced_dfs = []
    for cls in classes:
        cls_df = df[df[target_col] == cls]
        if len(cls_df) > target_size:
            cls_sampled = resample(cls_df, replace=False, n_samples=target_size, random_state=42)
        else:
            cls_sampled = resample(cls_df, replace=True, n_samples=target_size, random_state=42)
        balanced_dfs.append(cls_sampled)

    df_balanced = pd.concat(balanced_dfs)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced


# ==========================
# Load Data + Bag of Words
# ==========================
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df["clean_comment"] = df["clean_comment"].astype(str).apply(preprocess_comment)
    df = balance_dataset(df, 'category')

    # Bag Of Words instead of TF-IDF
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["clean_comment"])
    y = df["category"]

    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer


# ==========================
# Train & Log Model (Logistic Regression)
# ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):

    param_grid = {
        "C": [0.01, 0.1, 1],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear", "saga", "newton-cg"],
        "max_iter": [200, 300, 400]
    }

    with mlflow.start_run():
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        for params, mean_score, std_score in zip(grid_search.cv_results_["params"], 
                                                 grid_search.cv_results_["mean_test_score"], 
                                                 grid_search.cv_results_["std_test_score"]):

            with mlflow.start_run(run_name=f"LR BoW params: {params}", nested=True):
                model = LogisticRegression(**params, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
                    "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
                    "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }

                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f}")

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "bow_logistic_model")

        print(f"\n✅ Best Params: {best_params} | ✅ Best F1 Score: {best_f1:.4f}")


# ==========================
# Main
# ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data(
        "https://raw.githubusercontent.com/monalbhuva/Dataset/refs/heads/main/Reddit_Data.csv"
    )
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)
