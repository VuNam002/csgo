from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP_COLS = ["date", "day", "month", "year", "team_a_rounds", "team_b_rounds"]
NUM_COLS = ["wait_time_s", "match_time_s", "ping", "kills",
            "assists", "deaths", "mvps", "hs_percent", "points"]


def split_data(data, target="result", test_size=0.2, random_state=42):
    X = data.drop(columns=[target] + DROP_COLS)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


# Tiền xử lý dữ liệu
def create_preprocessing_pipeline():
    return ColumnTransformer(
        transformers=[
            ("map", OneHotEncoder(handle_unknown="ignore"), ["map"]),
            ("num", StandardScaler(), NUM_COLS),
        ],
        remainder="drop",
    )
