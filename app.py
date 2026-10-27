import streamlit as st
import pandas as pd

from modules.profiler import profile_dataset


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
