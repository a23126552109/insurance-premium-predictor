# Automated Insurance Premium Prediction Using Snowflake ML

## Project Overview

This project uses Machine Learning and Snowflake Cloud Data Platform to predict medical insurance premiums based on customer information.

An XGBoost Regression model is trained using historical insurance data, evaluated using standard regression metrics, and registered in Snowflake Model Registry.

A Streamlit application allows users to enter customer details, receive estimated insurance premiums, view analytics, and access prediction history.

The project also implements automated predictions using Snowflake Streams, Stored Procedures, and Tasks.

## Project Objectives

- Predict insurance premiums using Machine Learning.
- Store and process insurance data in Snowflake.
- Train and evaluate an XGBoost regression model.
- Register and manage the model using Snowflake Model Registry.
- Build an interactive Streamlit application.
- Automate predictions for newly inserted customer records.
- Store prediction results and timestamps in Snowflake.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Snowflake | Cloud data platform |
| Snowpark Python | Data processing |
| Snowflake ML | Model registry and inference |
| XGBoost | Regression model |
| Scikit-learn | Preprocessing and evaluation |
| Pandas and NumPy | Data manipulation |
| Streamlit | Interactive web application |
| SQL | Database operations and automation |

## Dataset

**Dataset:** Medical Cost Personal Dataset

**Source:** [Kaggle – Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

The dataset contains 1,338 records and 7 columns.

| Column | Description |
|---|---|
| AGE | Customer age |
| SEX | Customer gender |
| BMI | Body Mass Index |
| CHILDREN | Number of children |
| SMOKER | Smoking status |
| REGION | Residential region |
| CHARGES | Actual insurance charges |

**Target variable:** CHARGES

## Machine Learning Model

The project uses an XGBoost Regressor with a preprocessing pipeline.

### Data Preprocessing

- Numerical features: AGE, BMI, CHILDREN.
- Categorical features: SEX, SMOKER, REGION.
- OneHotEncoder is used for categorical features.
- Unknown categories are handled using `handle_unknown="ignore"`.

### Training Configuration

- Algorithm: XGBoost Regressor
- Number of estimators: 200
- Learning rate: 0.05
- Maximum depth: 3
- Random state: 42
- Training data: 80%
- Testing data: 20%

### Model Evaluation Results

| Metric | Result |
|---|---:|
| MAE | 2395.69 |
| RMSE | 4237.94 |
| R² Score | 0.8843 |
| R² Percentage | 88.43% |

The R² score indicates that the model explains approximately 88.43% of the variance in the test-set insurance charges. It is not classification accuracy.

## Snowflake Integration

### Snowflake Configuration

- Database: `INSURANCE_ML_DB`
- Schema: `INSURANCE_SCHEMA`
- Warehouse: `COMPUTE_WH`
- Model: `INSURANCE_CHARGES_XGB`
- Model Version: `V1`

### Main Snowflake Objects

| Object | Purpose |
|---|---|
| INSURANCE_DATA | Original insurance dataset |
| INSURANCE_PREDICTIONS | Test-set prediction results |
| NEW_INSURANCE_CUSTOMERS | New customer input records |
| INSURANCE_CUSTOMER_STREAM | Captures new customer changes |
| AUTOMATED_INSURANCE_PREDICTIONS | Stores automated predictions |
| PROCESS_INSURANCE_CUSTOMERS | Stored procedure for prediction processing |
| INSURANCE_PREDICTION_TASK | Scheduled automation task |

## Project Architecture

1. Historical insurance data is stored in Snowflake.
2. Data is loaded into Python using Snowpark.
3. Data is split into training and testing sets.
4. An XGBoost regression pipeline is trained.
5. The model is evaluated using MAE, RMSE and R².
6. The trained model is registered in Snowflake Model Registry.
7. Streamlit connects to Snowflake and requests predictions.
8. Prediction results are saved in Snowflake.
9. New customer records are captured by a Snowflake Stream.
10. A scheduled Task invokes a Stored Procedure to generate and save automated predictions.

## Streamlit Application Features

### 1. Premium Prediction

Users enter customer details and receive an estimated insurance premium.

### 2. Dashboard

Displays:
- Total predictions
- Average predicted premium
- Minimum and maximum premiums
- Prediction trends
- Customer analytics by gender, smoking status, region and age group
- Model evaluation metrics

### 3. Prediction History

Displays previously saved predictions and supports CSV export.

## Project Structure

```text
insurance-premium-predictor/
│
├── models/
│   ├── model_metrics.json
│   └── README.md
│
├── notebooks/
│   └── INSURANCE_PREDICTION_ML.ipynb
│
├── src/
│   ├── data_preprocess.py
│   ├── training.py
│   ├── prediction.py
│   └── streamlit_app.py
│
├── sql/
│   └── Insurance.sql
│
├── screenshots/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/insurance-premium-predictor.git
```

### 2. Open the Project

```bash
cd insurance-premium-predictor
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Snowflake Connection

Configure your Snowflake connection in the appropriate Streamlit or Snowflake deployment settings.

Provide the required account and authentication details securely.

Never upload passwords, tokens or `secrets.toml` to GitHub.

### 5. Run the Streamlit Application

```bash
streamlit run src/streamlit_app.py
```

Make sure your Snowflake database, schema, warehouse and registered model are available before running the application.

## Expected Output

The application generates an estimated insurance premium based on the customer's information.

Prediction results are stored in Snowflake and can be viewed through the dashboard and prediction history.

## Future Enhancements

- Compare multiple regression algorithms.
- Improve model performance through hyperparameter tuning.
- Add model monitoring and retraining.
- Add advanced customer analytics.
- Deploy the application for public access with secure authentication.

## Author

**Project:** Automated Insurance Premium Prediction Using Snowflake ML

**Domain:** Machine Learning | Cloud Computing | Data Analytics

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

**Note:** This project is intended for educational and demonstration purposes. Predicted premiums are estimates, not official insurance quotations.