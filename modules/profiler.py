import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Analyze a dataset and return its basic data-quality profile.
    """

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    # 1. Dataset dimensions
    rows, columns = df.shape

    # 2. Missing values
    missing_values = df.isnull().sum()

    missing_summary = {
        column: {
            "count": int(count),
            "percentage": round((count / rows) * 100, 2)
        }
        for column, count in missing_values.items()
    }

    # 3. Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # 4. Column data types
    data_types = {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }

    # 5. Unique values
    unique_counts = {
        column: int(df[column].nunique())
        for column in df.columns
    }

    # 6. Numerical summary statistics
    numerical_summary = df.describe(
        include="number"
    ).to_dict()

    # 7. Categorical summaries
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    categorical_summary = {}

    for column in categorical_columns:
        modes = df[column].mode()

        categorical_summary[column] = {
            "unique_values": int(df[column].nunique()),
            "most_frequent": (
                str(modes.iloc[0]) if not modes.empty else None
            )
        }

    return {
        "rows": rows,
        "columns": columns,
        "missing_values": missing_summary,
        "duplicate_rows": duplicate_count,
        "data_types": data_types,
        "unique_counts": unique_counts,
        "numerical_summary": numerical_summary,
        "categorical_summary": categorical_summary
    }
