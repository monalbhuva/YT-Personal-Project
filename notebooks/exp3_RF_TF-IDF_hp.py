import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Suppress MLflow artifact download warnings
# os.environ["MLFLOW_DISABLE_ARTIFACTS_DOWNLOAD"] = "1"

# Set MLflow Tracking URI & DAGsHub integration
MLFLOW_TRACKING_URI = "https://dagshub.com/monalAJ/YT-Personal-Project.mlflow"
dagshub.init(repo_owner="monalAJ", repo_name="YT-Personal-Project", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("RF Hyperparameter TuningV2")


# ==========================
# Text Preprocessing Functions
# ==========================
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


# ==========================
# Load & Prepare Data
# ==========================
def load_and_prepare_data(filepath):
    """Loads, preprocesses, and vectorizes the dataset."""
    df = pd.read_csv(filepath)
    
    # Apply text preprocessing
    df["clean_comment"] = df["clean_comment"].astype(str).apply(preprocess_comment)
    df = balance_dataset(df, 'category')
    # Convert text data to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_comment"])
    y = df["category"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer


# ==========================
# Train & Log Model
# ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""
    
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20]
    }
    
    with mlflow.start_run():
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Log all hyperparameter tuning runs
        for params, mean_score, std_score in zip(grid_search.cv_results_["params"], 
                                                 grid_search.cv_results_["mean_test_score"], 
                                                 grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"RF with params: {params}", nested=True):
                model = RandomForestClassifier(**params, random_state=42)
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

                
                # Log parameters & metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f}")

        # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"\nBest Params: {best_params} | Best F1 Score: {best_f1:.4f}")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data("https://raw.githubusercontent.com/monalbhuva/Dataset/refs/heads/main/Reddit_Data.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)