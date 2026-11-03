from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils.preprocessing import create_preprocessing_pipeline


def train_logistic_regression(X_train, X_test, y_train, y_test):
    prep = create_preprocessing_pipeline()
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    X_train_p = prep.fit_transform(X_train)
    clf.fit(X_train_p, y_train)

    return Pipeline([("prep", prep), ("clf", clf)])
