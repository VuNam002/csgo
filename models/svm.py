from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from utils.preprocessing import create_preprocessing_pipeline


def train_svm(X_train, X_test, y_train, y_test):
    prep = create_preprocessing_pipeline()
    clf = SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced",
        probability=True,
        random_state=42,
    )

    # Tập train: fit_transform = học tham số (mean, std, danh sách map) + biến đổi
    X_train_p = prep.fit_transform(X_train)
    clf.fit(X_train_p, y_train)

    return Pipeline([("prep", prep), ("clf", clf)])
