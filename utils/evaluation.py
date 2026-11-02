from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    # Tập test: chỉ transform (dùng tham số đã học từ train), KHÔNG fit lại
    X_test_p = prep.transform(X_test)
    y_pred = clf.predict(X_test_p)

    roc_auc = None
    if hasattr(clf, "predict_proba"):
        roc_auc = roc_auc_score(
            y_test,
            clf.predict_proba(X_test_p),
            multi_class="ovr",
            average="macro",
            labels=clf.classes_,
        )

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }
