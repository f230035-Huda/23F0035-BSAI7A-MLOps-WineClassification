import mlflow
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import load_and_split_data


def evaluate_registered_model():
    _, X_test, _, y_test = load_and_split_data()

    model_uri = "models:/WineClassifier@champion"
    model = mlflow.sklearn.load_model(model_uri)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    loss = log_loss(y_test, probabilities)

    print("\nInference Verification - Registered Champion Model")
    print(f"Model: {model_uri}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Log Loss: {loss:.4f}")

    return accuracy, macro_f1, loss


if __name__ == "__main__":
    evaluate_registered_model()
