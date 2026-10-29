import streamlit as st
import pandas as pd

from modules.profiler import profile_dataset
from modules.model_evaluator import evaluate_model
from modules.leakage_detector import detect_potential_leakage


st.set_page_config(
    page_title="LeakLens",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 LeakLens")
st.subheader("AI-Powered Machine Learning Audit Assistant")

st.write(
    "Analyze datasets, evaluate ML models, "
    "and investigate potential data leakage."
)

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV contains no data.")

        else:
            st.success("Dataset uploaded successfully!")
            st.subheader("7. ML Evaluation Setup")

            target_column = st.selectbox(
                "Select the target column",
                options=df.columns
            )

            st.write("Selected target:", target_column)
            st.write("Target data type:", df[target_column].dtype)
            st.write("Unique target values:", df[target_column].nunique())

# Suggest a task type based on the selected target
            target_dtype = df[target_column].dtype
            unique_values = df[target_column].nunique()

            if target_dtype == "object" or str(target_dtype) == "category" or target_dtype == "bool":
                suggested_task = "Classification"
            elif unique_values <= 10:
                suggested_task = "Classification"
            else:
                suggested_task = "Regression"

            st.subheader("ML Task Selection")

            task_type = st.selectbox(
                "Choose the machine learning task",
                options=["Classification", "Regression"],
                index=["Classification", "Regression"].index(suggested_task)
            )



            st.write("Suggested task:", suggested_task)
            st.write("Selected task:", task_type)
            st.subheader("Potential Data Leakage Detection")

            leakage_results = detect_potential_leakage(
                df=df,
                target_column=target_column
            )

            if leakage_results.empty:
                st.success(
                    "No suspicious features detected by the current checks. "
                    "This does not guarantee that the dataset is leakage-free."
                )
            else:
                st.warning(
                    f"Found {len(leakage_results)} suspicious feature(s) "
                    "that need investigation."
                )

                st.dataframe(
                    leakage_results,
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader("Baseline Model Evaluation")

            if st.button("Run Baseline Evaluation"):

                try:
                    with st.spinner("Training and evaluating the model..."):

                        result = evaluate_model(
                            df=df,
                            target_column=target_column,
                            task_type=task_type
                        )

                    st.success("Model evaluation completed!")

                    st.write("Training rows:", result["train_rows"])
                    st.write("Testing rows:", result["test_rows"])

                    if result["excluded_columns"]:
                        st.info(
                            "Excluded identifier-like columns: "
                            + ", ".join(result["excluded_columns"])
                        )

                    st.subheader("Evaluation Metrics")

                    metric_columns = st.columns(len(result["metrics"]))

                    for column, (metric_name, metric_value) in zip(
                        metric_columns, result["metrics"].items()
                    ):
                        column.metric(metric_name, f"{metric_value:.4f}")

                except Exception as error:
                    st.error(f"Evaluation failed: {error}")



            # Dataset preview
            st.subheader("1. Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Generate audit report
            profile = profile_dataset(df)

            # Dataset overview
            st.subheader("2. Dataset Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric("Rows", profile["rows"])
            col2.metric("Columns", profile["columns"])
            col3.metric("Duplicate Rows", profile["duplicate_rows"])

            # Missing values
            st.subheader("3. Missing Values")

            missing_df = pd.DataFrame.from_dict(
                profile["missing_values"],
                orient="index"
            )

            missing_df.index.name = "Feature"

            st.dataframe(missing_df, use_container_width=True)

            # Column data types
            st.subheader("4. Column Data Types")

            dtype_df = pd.DataFrame(
                list(profile["data_types"].items()),
                columns=["Feature", "Data Type"]
            )

            st.dataframe(dtype_df, use_container_width=True)

            # Numerical statistics
            st.subheader("5. Numerical Summary")

            if profile["numerical_summary"]:
                numerical_df = pd.DataFrame(
                    profile["numerical_summary"]
                )

                st.dataframe(
                    numerical_df,
                    use_container_width=True
                )
            else:
                st.info("No numerical columns found.")

            # Categorical statistics
            st.subheader("6. Categorical Summary")

            if profile["categorical_summary"]:
                categorical_df = pd.DataFrame.from_dict(
                    profile["categorical_summary"],
                    orient="index"
                )

                categorical_df.index.name = "Feature"

                st.dataframe(
                    categorical_df,
                    use_container_width=True
                )
            else:
                st.info("No categorical columns found.")

    except Exception as error:
        st.error(f"Unable to analyze this CSV: {error}")

else:
    st.info("Upload a CSV file to begin your dataset audit.")
