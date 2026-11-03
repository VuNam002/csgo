from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from utils.preprocessing import create_preprocessing_pipeline


def train_lightgbm(X_train, X_test, y_train, y_test):
    prep = create_preprocessing_pipeline()
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    X_train_p = prep.fit_transform(X_train)
    clf.fit(X_train_p, y_train)
    return Pipeline([("prep", prep), ("clf", clf)])

