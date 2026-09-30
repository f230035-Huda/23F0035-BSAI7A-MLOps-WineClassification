import time

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from src.data import load_and_split_data


def train_model():
    X_train, X_val, y_train, y_val = load_and_split_data()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_val, y_val


def test_validation_macro_f1_gate():
    model, X_val, y_val = train_model()

    predictions = model.predict(X_val)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    assert macro_f1 >= 0.88, (
        f"Validation Macro F1 {macro_f1:.4f} "
        f"is below the required 0.88"
    )


def test_inference_latency_gate():
    model, X_val, _ = train_model()

    start_time = time.perf_counter()
    model.predict(X_val)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    assert latency_ms <= 30, (
        f"Inference latency {latency_ms:.2f} ms "
        f"exceeds the 30 ms limit"
    )


def test_prediction_classes_gate():
    model, X_val, _ = train_model()

    predictions = model.predict(X_val)

    allowed_classes = {0, 1, 2}

    assert set(predictions).issubset(allowed_classes), (
        f"Unexpected prediction classes: {set(predictions)}"
    )
