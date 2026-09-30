from src.data import load_and_split_data
from src.data import validate_data


def test_data_split():
    X_train, X_val, y_train, y_val = load_and_split_data()
    assert X_train.shape == (142, 13)
    assert X_val.shape == (36, 13)
    validate_data(X_train, y_train)
    validate_data(X_val, y_val)


def test_classes():
    X_train, X_val, y_train, y_val = load_and_split_data()
    assert set(y_train) == {0, 1, 2}
    assert set(y_val) == {0, 1, 2}
