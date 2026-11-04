
import streamlit as st
import os

# =========================================
# 1. PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🛡️",
    layout="wide"
)

# =========================================
# 2. SNOWFLAKE CONNECTION
# =========================================

conn = st.connection(
    "snowflake",
    ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL")
)

session = conn.session()

# =========================================
# 3. HEADER
# =========================================

st.title("🛡️ Insurance Premium Predictor")

st.markdown(
    """
    ### AI-Powered Insurance Premium Estimation

    Predict insurance charges using our trained
    **XGBoost Machine Learning Model**.
    """
)

st.divider()

# =========================================
# 4. SIDEBAR NAVIGATION
# =========================================

st.sidebar.title("🛡️ Insurance ML")
st.sidebar.markdown("Navigate through the application")

page = st.sidebar.radio(
    "Select Page",
    [
        "Premium Prediction",
        "Dashboard",
        "Prediction History"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Powered by Snowflake ML + XGBoost")


# =========================================
# 5. PREMIUM PREDICTION PAGE
# =========================================

if page == "Premium Prediction":

    st.header("👤 Enter Customer Details")

    with st.form("insurance_form"):

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=29
            )

            sex = st.selectbox(
                "Gender",
                ["male", "female"]
            )

            bmi = st.number_input(
                "Body Mass Index (BMI)",
                min_value=10.0,
                max_value=60.0,
                value=28.4,
                step=0.1
            )

        with col2:

            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=10,
                value=1
            )

            smoker = st.selectbox(
                "Smoking Status",
                ["no", "yes"]
            )

            region = st.selectbox(
                "Residential Region",
                [
                    "northeast",
                    "northwest",
                    "southeast",
                    "southwest"
                ]
            )

        submitted = st.form_submit_button(
            "🔍 Predict Insurance Premium",
            use_container_width=True
        )

    # Process prediction after form submission

    if submitted:

        try:

            with st.spinner("🤖 AI model is predicting..."):

                # ---------------------------------
                # CALL REGISTERED XGBOOST MODEL
                # ---------------------------------

                prediction_result = session.sql(
                    """
                    SELECT
                        (
                            INSURANCE_ML_DB.INSURANCE_SCHEMA
                            .INSURANCE_CHARGES_XGB!PREDICT(
                                ?, ?, ?, ?, ?, ?
                            )
                        ):"output_feature_0"::FLOAT
                        AS PREDICTED_CHARGES
                    """,
                    params=[
                        int(age),
                        sex,
                        float(bmi),
                        int(children),
                        smoker,
                        region
                    ]
                ).collect()

                # Extract numeric prediction directly
                predicted_charges = float(
                    prediction_result[0]["PREDICTED_CHARGES"]
                )

                # ---------------------------------
                # SAVE PREDICTION TO SNOWFLAKE
                # ---------------------------------

                session.sql(
                    """
                    INSERT INTO
                    INSURANCE_ML_DB.INSURANCE_SCHEMA
                    .AUTOMATED_INSURANCE_PREDICTIONS
                    (
                        AGE,
                        SEX,
                        BMI,
                        CHILDREN,
                        SMOKER,
                        REGION,
                        PREDICTED_CHARGES,
                        PREDICTION_DATE
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?,
                        CURRENT_TIMESTAMP()::TIMESTAMP_NTZ
                    )
                    """,
                    params=[
                        int(age),
                        sex,
                        float(bmi),
                        int(children),
                        smoker,
                        region,
                        predicted_charges
                    ]
                ).collect()

            # ---------------------------------
            # DISPLAY RESULT
            # ---------------------------------

            st.success("✅ Prediction generated successfully!")

            st.divider()

            st.subheader("🎯 Prediction Result")

            st.metric(
                label="Estimated Insurance Premium",
                value=f"${predicted_charges:,.2f}"
            )

            st.info(
                "Your prediction has been successfully "
                "saved in the Snowflake database."
            )

            # Customer summary

            st.subheader("Customer Information")

            result_col1, result_col2, result_col3 = st.columns(3)

            result_col1.write(f"**Age:** {age}")
            result_col1.write(f"**Gender:** {sex}")

            result_col2.write(f"**BMI:** {bmi}")
            result_col2.write(f"**Children:** {children}")

            result_col3.write(f"**Smoker:** {smoker}")
            result_col3.write(f"**Region:** {region}")

        except Exception as e:

            st.error(f"Prediction failed: {str(e)}")


# =========================================
# 6. DASHBOARD PAGE
# =========================================

elif page == "Dashboard":

    st.header("📊 Insurance Prediction Dashboard")

    try:

        dashboard = session.sql(
            """
            SELECT
                COUNT(*) AS TOTAL_CUSTOMERS,
                ROUND(AVG(PREDICTED_CHARGES), 2)
                    AS AVG_PREMIUM,
                ROUND(MIN(PREDICTED_CHARGES), 2)
                    AS MIN_PREMIUM,
                ROUND(MAX(PREDICTED_CHARGES), 2)
                    AS MAX_PREMIUM
            FROM
                INSURANCE_ML_DB.INSURANCE_SCHEMA
                .AUTOMATED_INSURANCE_PREDICTIONS
            """
        ).to_pandas()

        total = int(dashboard.iloc[0]["TOTAL_CUSTOMERS"])

        if total == 0:

            st.info("No prediction records available yet.")

        else:

            avg = dashboard.iloc[0]["AVG_PREMIUM"]
            minimum = dashboard.iloc[0]["MIN_PREMIUM"]
            maximum = dashboard.iloc[0]["MAX_PREMIUM"]

            # Dashboard metrics

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            col1.metric(
                "Total Predictions",
                f"{total:,}"
            )

            col2.metric(
                "Average Premium",
                f"${avg:,.2f}"
            )

            col3.metric(
                "Minimum Premium",
                f"${minimum:,.2f}"
            )

            col4.metric(
                "Maximum Premium",
                f"${maximum:,.2f}"
            )

            st.divider()

            # Prediction chart

            st.subheader("📈 Prediction Overview")

            history = session.sql(
                """
                SELECT
                    PREDICTION_DATE,
                    PREDICTED_CHARGES
                FROM
                    INSURANCE_ML_DB.INSURANCE_SCHEMA
                    .AUTOMATED_INSURANCE_PREDICTIONS
                ORDER BY PREDICTION_DATE
                """
            ).to_pandas()

            if not history.empty:

                history["PREDICTION_DATE"] = (
                    history["PREDICTION_DATE"].astype(str)
                )

                st.line_chart(
                    history.set_index("PREDICTION_DATE")[
                        "PREDICTED_CHARGES"
                    ]
                )

    except Exception as e:

        st.error(f"Dashboard error: {str(e)}")


# =========================================
# 7. PREDICTION HISTORY PAGE
# =========================================

elif page == "Prediction History":

    st.header("📋 Customer Prediction History")

    try:

        history = session.sql(
            """
            SELECT
                AGE,
                SEX,
                BMI,
                CHILDREN,
                SMOKER,
                REGION,
                ROUND(PREDICTED_CHARGES, 2)
                    AS PREDICTED_CHARGES,
                PREDICTION_DATE
            FROM
                INSURANCE_ML_DB.INSURANCE_SCHEMA
                .AUTOMATED_INSURANCE_PREDICTIONS
            ORDER BY PREDICTION_DATE DESC
            """
        ).to_pandas()

        if history.empty:

            st.info("No prediction history available.")

        else:

            st.success(
                f"Total prediction records: {len(history)}"
            )

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

            # Download CSV

            st.download_button(
                label="📥 Download Prediction History",
                data=history.to_csv(index=False),
                file_name="insurance_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(f"History error: {str(e)}")


# =========================================
# 8. FOOTER
# =========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <p>🛡️ <b>Automated Insurance Premium Prediction</b></p>
        <p>Machine Learning Model: XGBoost</p>
        <p>Database: Snowflake</p>
        <p>Developed using Python and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
