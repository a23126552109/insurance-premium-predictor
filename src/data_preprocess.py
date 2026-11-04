"""Data loading and preprocessing utilities for the insurance premium project."""

from sklearn.model_selection import train_test_split

DATABASE = "INSURANCE_ML_DB"
SCHEMA = "INSURANCE_SCHEMA"
SOURCE_TABLE = f"{DATABASE}.{SCHEMA}.INSURANCE_DATA"

NUMERIC_FEATURES = ["AGE", "BMI", "CHILDREN"]
CATEGORICAL_FEATURES = ["SEX", "SMOKER", "REGION"]
TARGET_COLUMN = "CHARGES"


def load_insurance_data(session, table_name=SOURCE_TABLE):
    """Load the Snowflake insurance table into a pandas DataFrame."""
    insurance_df = session.table(table_name).to_pandas()

    if insurance_df.empty:
        raise ValueError(f"No records found in {table_name}")

    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in insurance_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return insurance_df


def split_features_target(insurance_df, test_size=0.2, random_state=42):
    """Separate features/target and create the notebook's 80:20 split."""
    X = insurance_df.drop(TARGET_COLUMN, axis=1)
    y = insurance_df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
