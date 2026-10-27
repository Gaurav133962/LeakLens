import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(df, target_column, task_type):

    # 1. Remove rows where the target is missing
    df = df.dropna(subset=[target_column]).copy()

    # 2. Separate input features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 3. Exclude identifier-like columns for this baseline
    id_columns = [
        col for col in X.columns
        if col.lower() in [
            "id", "passengerid", "customerid",
            "transactionid", "userid"
        ]
    ]

    X = X.drop(columns=id_columns)

    if X.shape[1] == 0:
        raise ValueError(
            "No usable input features remain after excluding identifier columns."
        )

    # 4. Identify numerical and categorical features
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # 5. Create preprocessing pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 6. Apply the appropriate preprocessing to each column type
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # 7. Select a baseline model
    if task_type == "Classification":
        if y.nunique() < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )

        model = LogisticRegression(max_iter=1000)

    elif task_type == "Regression":
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                "Regression requires a numerical target column."
            )

        model = LinearRegression()

    else:
        raise ValueError("Unsupported machine learning task.")

    # 8. Combine preprocessing and model into one pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # 9. Split the data before fitting preprocessing
    stratify_values = None

    if task_type == "Classification":
        class_counts = y.value_counts()

        if len(class_counts) >= 2 and class_counts.min() >= 2:
            stratify_values = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_values
    )

    # 10. Train the complete pipeline
    pipeline.fit(X_train, y_train)

    # 11. Predict using unseen test data
    predictions = pipeline.predict(X_test)

    # 12. Calculate evaluation metrics
    if task_type == "Classification":

        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, average="weighted", zero_division=0
            )
        }

    else:

        metrics = {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(
                y_test, predictions
            ) ** 0.5,
            "R2 Score": r2_score(y_test, predictions)
        }

    return {
        "metrics": metrics,
        "model": pipeline,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "excluded_columns": id_columns,
        "feature_columns": X.columns.tolist()
    }
