import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.data import load_and_split_data

RANDOM_STATE = 42
EXPERIMENT_NAME = "Wine-Cultivar-Classification"


def get_model_configs():
    random_forest_configs = {
        "rf_1": RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=RANDOM_STATE),
        "rf_2": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
        "rf_3": RandomForestClassifier(
            n_estimators=300, max_depth=5, random_state=RANDOM_STATE)
    }
    gradient_boosting_configs = {
        "gb_1": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=RANDOM_STATE),
        "gb_2": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE),
        "gb_3": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=2,
            random_state=RANDOM_STATE)
    }
    return {
        "RandomForest": random_forest_configs,
        "GradientBoosting": gradient_boosting_configs
    }


def cross_validate_models(X_train, y_train):
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "neg_log_loss": "neg_log_loss"
    }
    results = []
    for model_family, configs in get_model_configs().items():
        for config_name, model in configs.items():
            scores = cross_validate(
                model, X_train, y_train, cv=cv, scoring=scoring,
                return_train_score=True)
            result = {
                "model_family": model_family,
                "config": config_name,
                "train_accuracy": scores["train_accuracy"].mean(),
                "validation_accuracy": scores["test_accuracy"].mean(),
                "train_macro_f1": scores["train_macro_f1"].mean(),
                "validation_macro_f1": scores["test_macro_f1"].mean(),
                "train_log_loss": -scores["train_neg_log_loss"].mean(),
                "validation_log_loss": -scores["test_neg_log_loss"].mean()
            }
            results.append(result)
    return results


def log_mlflow_run(model_family, config_name, model, result, X_train):
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{model_family}_{config_name}"):
        mlflow.log_params(model.get_params())
        mlflow.set_tag("model_family", model_family)
        mlflow.set_tag("configuration", config_name)
        mlflow.log_metric("validation_accuracy", result["validation_accuracy"])
        mlflow.log_metric("validation_macro_f1", result["validation_macro_f1"])
        mlflow.log_metric("validation_log_loss", result["validation_log_loss"])
        mlflow.log_metric("train_macro_f1", result["train_macro_f1"])
        input_example = X_train[:1]
        signature = infer_signature(
            X_train, model.predict(X_train)
        )
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example
        )


def register_best_model(results):
    best_result = max(
        results,
        key=lambda x: (
            x["validation_macro_f1"],
            -x["validation_log_loss"]
        )
    )
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.configuration = '{best_result['config']}' "
            f"and tags.model_family = '{best_result['model_family']}'"
        )
    )
    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="WineClassifier"
    )
    client.set_registered_model_alias(
        "WineClassifier",
        "champion",
        registered_model.version
    )
    print(
        f"\nBest Model: {best_result['model_family']} | "
        f"{best_result['config']}"
    )
    print(
        f"Validation Macro F1: "
        f"{best_result['validation_macro_f1']:.4f}"
    )
    print(
        f"Registered as WineClassifier version "
        f"{registered_model.version}"
    )
    print("Alias: champion")


def evaluate_registered_model(X_val, y_val):
    model_uri = "models:/WineClassifier@champion"
    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)
    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(y_val, predictions, average="macro")
    loss = log_loss(y_val, probabilities)
    print("\nRegistered Champion Model Evaluation")
    print(f"Model: {model_uri}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Log Loss: {loss:.4f}")


def evaluate_on_validation(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)
    return {
        "accuracy": accuracy_score(y_val, predictions),
        "macro_f1": f1_score(y_val, predictions, average="macro"),
        "log_loss": log_loss(y_val, probabilities)
    }


def main():
    X_train, X_val, y_train, y_val = load_and_split_data()
    results = cross_validate_models(X_train, y_train)
    print("\n5-Fold Stratified Cross-Validation Results")
    for result in results:
        print(
            f"{result['model_family']} | {result['config']} | "
            f"Train F1: {result['train_macro_f1']:.4f} | "
            f"Val F1: {result['validation_macro_f1']:.4f} | "
            f"Val Accuracy: {result['validation_accuracy']:.4f} | "
            f"Val Log Loss: {result['validation_log_loss']:.4f}"
        )
    print("\nLogging MLflow Runs")
    for model_family, configs in get_model_configs().items():
        for config_name, model in configs.items():
            result = next(
                item for item in results
                if item["model_family"] == model_family
                and item["config"] == config_name
            )
            model.fit(X_train, y_train)
            log_mlflow_run(
                model_family, config_name, model, result, X_train)
            print(f"Logged: {model_family} | {config_name}")
    register_best_model(results)
    print("\nHoldout Validation Results")
    for model_family, configs in get_model_configs().items():
        for config_name, model in configs.items():
            metrics = evaluate_on_validation(
                model, X_train, y_train, X_val, y_val)
            print(
                f"{model_family} | {config_name} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"Macro F1: {metrics['macro_f1']:.4f} | "
                f"Log Loss: {metrics['log_loss']:.4f}"
            )
    evaluate_registered_model(X_val, y_val)


if __name__ == "__main__":
    main()
