import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"


def load_and_split_data(test_size=0.2, random_state=42):

    df = pd.read_csv(URL)

    # Separate target
    y = df["median_house_value"]
    X = df.drop(columns=["median_house_value"])

    # --- One-Hot Encode categorical column ---
    X = pd.get_dummies(X, columns=["ocean_proximity"], drop_first=True)

    # --- Train Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
