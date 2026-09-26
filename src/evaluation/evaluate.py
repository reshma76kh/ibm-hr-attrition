import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_models(
    trained_models,
    X_test,
    y_test
):
    """
    Evaluate all trained models on the untouched test data.
    """

    results = []

    for name, model in trained_models.items():

        print(f"Evaluating {name}...")

        # Predictions
        y_pred = model.predict(X_test)

        # Tuned models were trained using:
        # No -> 0
        # Yes -> 1
        # Convert predictions back to original labels
        y_pred = pd.Series(y_pred).map({
            0: "No",
            1: "Yes"
        }).values

        # Probability for ROC-AUC
        if hasattr(model, "predict_proba"):

            y_probability = model.predict_proba(X_test)[:, 1]

        else:

            y_probability = model.decision_function(X_test)

        # Convert actual target to binary
        y_test_binary = y_test.map({
            "No": 0,
            "Yes": 1
        })

        # Metrics
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            pos_label="Yes",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label="Yes",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label="Yes",
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test_binary,
            y_probability
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=["No", "Yes"]
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "Confusion Matrix": cm
        })

    results_df = pd.DataFrame(results)

    return results_df