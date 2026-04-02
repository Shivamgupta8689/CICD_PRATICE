import mlflow
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set same experiment
mlflow.set_experiment("churn-prediction")

def evaluate():
    # Load test data
    df = pd.read_csv("data/processed/test.csv")
    X_test = df.drop("target", axis=1)
    y_test = df["target"]

    # Load trained model
    model = pickle.load(open("models/model.pkl", "rb"))

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

    print("✅ Evaluation Complete")
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


if __name__ == "__main__":
    evaluate()