
import pandas as pd


def detect_potential_leakage(df, target_column):
    """
    Identify suspicious features that may indicate data leakage.
    These checks flag candidates for investigation, not confirmed leakage.
    """

    if target_column not in df.columns:
        raise ValueError("Target column not found in dataset.")

    findings = []

    target = df[target_column]

    for column in df.columns:
        if column == target_column:
            continue

        feature = df[column]
        reasons = []

        # Check 1: Exact copies of the target
        comparison = pd.DataFrame({
            "feature": feature,
            "target": target
        }).dropna()

        if not comparison.empty:
            if comparison["feature"].equals(comparison["target"]):
                reasons.append(
                    "Feature exactly matches the target on non-missing rows."
                )

        # Check 2: Suspicious column names
        suspicious_names = [
            "target", "outcome", "label", "prediction",
            "result", "final_status", "actual_result"
        ]

        name = column.lower().replace(" ", "_")

        if name in suspicious_names:
            reasons.append(
                "Column name may indicate a target-derived value."
            )

        # Check 3: Very high numeric correlation
        if (
            pd.api.types.is_numeric_dtype(feature)
            and pd.api.types.is_numeric_dtype(target)
        ):
            numeric_data = pd.DataFrame({
                "feature": feature,
                "target": target
            }).dropna()

            if (
                len(numeric_data) >= 3
                and numeric_data["feature"].nunique() > 1
                and numeric_data["target"].nunique() > 1
            ):
                correlation = numeric_data["feature"].corr(
                    numeric_data["target"]
                )

                if abs(correlation) >= 0.98:
                    reasons.append(
                        f"Very high absolute correlation with target "
                        f"({correlation:.3f})."
                    )

        if reasons:
            findings.append({
                "Feature": column,
                "Warning": "Potential leakage or suspicious feature",
                "Evidence": " ".join(reasons)
            })

    return pd.DataFrame(
        findings,
        columns=["Feature", "Warning", "Evidence"]
    )
