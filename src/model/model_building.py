# src/model/model_building.py

import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_bow(train_data: pd.DataFrame) -> tuple:
    """Apply Bag-of-Words transformation."""
    try:
        vectorizer = CountVectorizer()

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        X_train_bow = vectorizer.fit_transform(X_train)

        logger.debug(f"BoW created with shape: {X_train_bow.shape}")

        # Save vectorizer
        with open(os.path.join(get_root_directory(), 'bow_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug("BoW vectorizer saved successfully")

        return X_train_bow, y_train
    except Exception as e:
        logger.error(f"Error during BoW transformation: {e}")
        raise



def train_logistic_regression(X_train, y_train, c_value: float, max_iter: int):
    """Train Logistic Regression classifier."""
    try:
        model = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            solver='newton-cg',
            penalty = 'l2'
        )
        model.fit(X_train, y_train)

        logger.debug("Logistic Regression model training completed successfully")
        return model
    except Exception as e:
        logger.error(f"Error training Logistic Regression: {e}")
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        root_dir = get_root_directory()

        # Load params
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        c_value = params['model_building']['C']
        max_iter = params['model_building']['max_iter']

        c_value = 1
        max_iter = 200

        # Load processed training data
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Apply Bag of Words
        X_train_bow, y_train = apply_bow(train_data)

        # Train Logistic Regression
        model = train_logistic_regression(X_train_bow, y_train, c_value, max_iter)

        # Save final trained model
        save_model(model, os.path.join(root_dir, 'logreg_bow_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
