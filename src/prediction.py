"""Prediction utilities for the registered Snowflake insurance model."""

import pandas as pd

from data_preprocess import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)

MODEL_NAME = "INSURANCE_CHARGES_XGB"
MODEL_VERSION = "V1"


def load_registered_model(session, model_name=MODEL_NAME, version_name=MODEL_VERSION):
    """Load a model version from the Snowflake Model Registry."""
    from snowflake.ml.registry import Registry

    registry = Registry(
        session=session,
        database_name="INSURANCE_ML_DB",
        schema_name="INSURANCE_SCHEMA"
    )

    registered_model = registry.get_model(model_name)
    return registered_model.version(version_name)


def predict_with_registered_model(model_version, customer_data):
    """
    Predict charges for a pandas DataFrame containing AGE, SEX, BMI,
    CHILDREN, SMOKER and REGION.
    """
    feature_columns = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    missing = [
        column for column in feature_columns
        if column not in customer_data.columns
    ]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    input_df = customer_data[feature_columns].copy()
    result = model_version.run(input_df, function_name="predict")

    if "output_feature_0" not in result.columns:
        raise ValueError(
            "The registered model output did not contain 'output_feature_0'. "
            f"Received columns: {list(result.columns)}"
        )

    output = input_df.copy()
    output["PREDICTED_CHARGES"] = result["output_feature_0"].values
    return output


def predict_one(model_version, age, sex, bmi, children, smoker, region):
    """Predict the premium for one customer and return a float."""
    customer = pd.DataFrame([{
        "AGE": int(age),
        "SEX": str(sex).lower(),
        "BMI": float(bmi),
        "CHILDREN": int(children),
        "SMOKER": str(smoker).lower(),
        "REGION": str(region).lower()
    }])

    result = predict_with_registered_model(model_version, customer)
    return float(result.loc[0, "PREDICTED_CHARGES"])
