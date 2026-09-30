from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_and_split_data():
    wine = load_wine()
    X = wine.data
    y = wine.target
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train, X_val, y_train, y_val


def validate_data(X, y):
    assert X.shape[1] == 13, "Dataset must contain 13 features."
    assert len(X) == len(y), "Features and labels must have equal length."
    assert not np.isnan(X).any(), "Features contain NaN values."
    assert not np.isnan(y).any(), "Labels contain NaN values."
