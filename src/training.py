"""XGBoost model creation, training, evaluation and Snowflake registration."""

import numpy as np

from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_model():
    """Build the same preprocessing + XGBoost pipeline used in the notebook."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                    objective="reg:squarederror"
                )
            )
        ]
    )


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train the pipeline and return the model, predictions and metrics."""
    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2_SCORE": float(r2)
    }

    return model, y_pred, metrics


def register_model(session, model, X_train, metrics,
                   model_name="INSURANCE_CHARGES_XGB",
                   version_name="V1"):
    """Log the trained pipeline in the Snowflake Model Registry."""
    from snowflake.ml.registry import Registry
    from snowflake.ml.model import task

    registry = Registry(
        session=session,
        database_name="INSURANCE_ML_DB",
        schema_name="INSURANCE_SCHEMA"
    )

    model_version = registry.log_model(
        model=model,
        model_name=model_name,
        version_name=version_name,
        sample_input_data=X_train,
        metrics=metrics,
        comment="XGBoost insurance premium prediction model",
        task=task.Task.TABULAR_REGRESSION
    )

    return model_version
