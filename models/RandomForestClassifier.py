from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from utils.preprocessing import create_preprocessing_pipeline


def train_random_forest(X_train, X_test, y_train, y_test):
    prep = create_preprocessing_pipeline()
    clf = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        class_weight="balanced",
        random_state=42,
    )

    X_train_p = prep.fit_transform(X_train)
    clf.fit(X_train_p, y_train)

    return Pipeline([("prep", prep), ("clf", clf)])
