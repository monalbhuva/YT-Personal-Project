import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "https://raw.githubusercontent.com/monalbhuva/Dataset/refs/heads/main/Reddit_Data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/monalAJ/YT-Personal-Project.mlflow",
    "dagshub_repo_owner": "monalAJ",
    "dagshub_repo_name": "YT-Personal-Project",
    "experiment_name": "Ex-2 : Bow vs TfIdf"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== BALANCE DATASET ==========================
from sklearn.utils import resample

def balance_dataset(df, target_col='category'):
    """
    Balances a multiclass dataset using hybrid over- and under-sampling.
    Returns a shuffled balanced DataFrame.
    """
    try:
        classes = df[target_col].unique()
        # Determine target size: average of all classes
        target_size = int(df[target_col].value_counts().mean())
        print("Target size per class:", target_size)
        
        balanced_dfs = []
        for cls in classes:
            cls_df = df[df[target_col] == cls]
            if len(cls_df) > target_size:
                # Undersample majority class
                cls_sampled = resample(cls_df, replace=False, n_samples=target_size, random_state=42)
            else:
                # Oversample minority class
                cls_sampled = resample(cls_df, replace=True, n_samples=target_size, random_state=42)
            balanced_dfs.append(cls_sampled)
        
        # Combine all classes
        df_balanced = pd.concat(balanced_dfs)
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Check distribution
        print("Balanced class distribution:\n", df_balanced[target_col].value_counts())
        return df_balanced
    
    except Exception as e:
        print(f"Error in balancing dataset: {e}")
        raise

# ========================== TEXT PREPROCESSING ==========================
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment
    

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        print('Text normalization completed')
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise

# ========================== LOAD & PREPROCESS DATA ==========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['clean_comment'] = df['clean_comment'].astype(str)
        df = normalize_text(df)
        df = balance_dataset(df, 'category')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
def map_labels_for_xgboost(y):
    """
    Maps labels from [-1, 0, 1] → [0, 1, 2] for XGBoost compatibility.
    Returns mapped labels and the mapping dictionary.
    """
    label_mapping = {-1: 0, 0: 1, 1: 2}
    y_mapped = y.map(label_mapping)
    return y_mapped, label_mapping

# ========================== FEATURE ENGINEERING ==========================
VECTORIZERS = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'LightGBM': LGBMClassifier()  
}

# ========================== TRAIN & EVALUATE MODELS ==========================
def train_and_evaluate(df):
    with mlflow.start_run(run_name="All Experiments") as parent_run:
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                    try:
                        # Feature extraction
                        X = vectorizer.fit_transform(df['clean_comment'])
                        y = df['category']
                        # For XGBoost, map labels to non-negative integers
                        if algo_name == "LightGBM":
                            X = X.astype(np.float32)
                        if algo_name in ["XGBoost", "LightGBM"]:
                            y, label_mapping = map_labels_for_xgboost(y)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)

                        # Log preprocessing parameters
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })

                        # Train model
                        model = algorithm
                        model.fit(X_train, y_train)

                        # Log model parameters
                        log_model_params(algo_name, model)

                        # Evaluate model
                        y_pred = model.predict(X_test)
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred,average="weighted"),
                            "recall": recall_score(y_test, y_pred,average="weighted"),
                            "f1_score": f1_score(y_test, y_pred,average="weighted")
                        }
                        mlflow.log_metrics(metrics)

                        # Log model
                        # mlflow.sklearn.log_model(model, "model")
                        input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                        mlflow.sklearn.log_model(model, "model", input_example=input_example)

                        # Print results for verification
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")

                    except Exception as e:
                        print(f"Error in training {algo_name} with {vec_name}: {e}")
                        mlflow.log_param("error", str(e))

def log_model_params(algo_name, model):
    """Logs hyperparameters of the trained model to MLflow."""
    params_to_log = {}
    if algo_name == 'LogisticRegression':
        params_to_log["C"] = model.C
    elif algo_name == 'MultinomialNB':
        params_to_log["alpha"] = model.alpha
    elif algo_name == 'XGBoost':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
    elif algo_name == 'RandomForest':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoosting':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'LightGBM':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth


    mlflow.log_params(params_to_log)

# ========================== EXECUTION ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)