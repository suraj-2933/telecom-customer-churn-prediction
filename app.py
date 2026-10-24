import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load saved files
model = joblib.load(r"C:\Users\Lenovo\Downloads\churn_pred_model.pkl")
columns = joblib.load(r"C:\Users\Lenovo\Downloads\columns (3).pkl")
encode_columns = joblib.load(r"C:\Users\Lenovo\Downloads\encode_columns (2).pkl")

# 1. Configure the page — this must come before other Streamlit UI commands
st.set_page_config(
    page_title="Telecom Insights",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS: improve spacing, colors, cards and buttons
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    [data-testid="stMetric"] {
        background-color: #f1f5f9;
        border: 1px solid #dbe4f0;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #475569;
        font-size: 14px;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
    }

    h1, h2, h3 {
        color: #0f172a;
    }

    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load the dataset
df = pd.read_excel(
    r"C:\Users\Lenovo\Downloads\telecom_customer_churn.xlsx"
)

# 4. Prepare the basic churn counts
status_column = "Customer Status"

if status_column not in df.columns:
    st.error("The dataset does not contain a 'Customer Status' column.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

status = df[status_column].astype(str).str.strip().str.lower()

churn_count = int((status == "churned").sum())
stayed_count = int((status == "stayed").sum())
total_customers = len(df)

churn_rate = (
    churn_count / total_customers * 100
    if total_customers > 0 else 0
)

# 5. Sidebar navigation
st.sidebar.title("📡 Telecom Insights")
st.sidebar.caption("Customer intelligence platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Churn Analytics", "Customer Data", "Prediction", "About Model"]
)

st.sidebar.divider()
st.sidebar.caption("Telecom Customer Churn Project")

# 6. Overview page
if page == "Overview":

    st.title("Customer Overview")
    st.markdown(
        "Understand customer retention, churn patterns and business performance."
    )
    st.divider()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{total_customers:,}")

    with col2:
        st.metric("Customers Churned", f"{churn_count:,}")

    with col3:
        st.metric("Customers Stayed", f"{stayed_count:,}")

    with col4:
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    st.write("")

    # Main charts
    left, right = st.columns(2)

    with left:
        st.subheader("Customer Retention")

        retention_df = pd.DataFrame({
            "Customer Status": ["Churned", "Stayed"],
            "Customers": [churn_count, stayed_count]
        })

        fig = px.pie(
            retention_df,
            names="Customer Status",
            values="Customers",
            hole=0.58,
            color="Customer Status",
            color_discrete_map={
                "Churned": "#ef4444",
                "Stayed": "#2563eb"
            }
        )

        fig.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            legend_title_text="Status"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Customer Status Distribution")

        fig = px.bar(
            retention_df,
            x="Customer Status",
            y="Customers",
            color="Customer Status",
            text="Customers",
            color_discrete_map={
                "Churned": "#ef4444",
                "Stayed": "#2563eb"
            }
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Number of customers",
            margin=dict(t=20, b=20, l=10, r=10)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business insight: churn rate shows the share of customers "
        "who left. Later, we will identify which customer groups are "
        "most likely to churn."
    )

# 7. Churn analytics page
elif page == "Churn Analytics":

    st.title("Churn Analytics")
    st.caption("Explore which customer groups may be associated with churn.")
    st.divider()

    possible_columns = [
        col for col in
        ["Contract", "Internet Type", "Payment Method", "Tenure in Months"]
        if col in df.columns
    ]

    if not possible_columns:
        st.warning("No supported analysis columns were found in this dataset.")
    else:
        selected_column = st.selectbox(
            "Choose a customer attribute",
            possible_columns
        )

        chart_df = df.copy()
        chart_df["_Status"] = (
            chart_df[status_column].astype(str).str.strip().str.lower()
        )
        chart_df = chart_df[
            chart_df["_Status"].isin(["churned", "stayed"])
        ]

        if selected_column == "Tenure in Months":
            chart_df[selected_column] = pd.to_numeric(
                chart_df[selected_column], errors="coerce"
            )
            chart_df = chart_df.dropna(subset=[selected_column])
            chart_df["Tenure Group"] = pd.cut(
                chart_df[selected_column],
                bins=[-1, 12, 24, 48, float("inf")],
                labels=["0–12 months", "13–24 months",
                        "25–48 months", "49+ months"]
            )
            group_column = "Tenure Group"
        else:
            group_column = selected_column

        if chart_df[group_column].notna().any():
            summary = (
                chart_df.groupby(
                    [group_column, "_Status"],
                    observed=False
                )
                .size()
                .reset_index(name="Customers")
            )

            fig = px.bar(
                summary,
                x=group_column,
                y="Customers",
                color="_Status",
                barmode="group",
                color_discrete_map={
                    "churned": "#ef4444",
                    "stayed": "#2563eb"
                },
                labels={"_Status": "Customer Status"}
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("There is no usable data for this attribute.")

# 8. Customer dataset page
elif page == "Customer Data":

    st.title("Customer Data Explorer")
    st.caption("Inspect the dataset used for churn analysis.")
    st.divider()

    search = st.text_input("Search across customer records")

    display_df = df.copy()

    if search:
        matches = display_df.astype(str).apply(
            lambda column: column.str.contains(
                search, case=False, na=False
            )
        ).any(axis=1)

        display_df = display_df[matches]

    st.write(f"Showing **{len(display_df):,}** records")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download displayed data as CSV",
        data=csv_data,
        file_name="telecom_customer_data.csv",
        mime="text/csv"
    )

# 9. Prediction page — UI placeholder for the next step

elif page == "Prediction":

    st.title("🔮 Customer Churn Prediction")
    st.markdown(
        "Enter the customer's details to estimate their likelihood of leaving."
    )
    st.divider()

    # Customer input form
    with st.form("churn_prediction_form"):

        st.subheader("👤 Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input(
                "Age", min_value=18, max_value=100, value=35
            )
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.number_input(
                "Number of Dependents", min_value=0,max_value=3 ,value=0
            )

        with col2:
            referrals = st.number_input(
                "Number of Referrals", min_value=0,max_value=11 ,value=0
            )
            tenure = st.number_input(
                "Tenure in Months", min_value=0, max_value=100, value=12
            )
            contract = st.selectbox(
                "Contract", ["Month-to-Month", "One Year", "Two Year"]
            )
            offer = st.selectbox(
                "Offer",
                ["No Offer", "Offer A", "Offer B", "Offer C",
                 "Offer D", "Offer E"]
            )

        with col3:
            zip_code = st.number_input(
                "Zip Code", min_value=0, value=90001
            )
            latitude = st.number_input("Latitude", value=33.0)
            longitude = st.number_input("Longitude", value=-117.0)

        st.divider()
        st.subheader("📱 Phone and Internet Services")

        col1, col2, col3 = st.columns(3)

        with col1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            long_distance = st.number_input(
                "Average Monthly Long Distance Charges",
                min_value=0.0, value=20.0
            )
            multiple_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No"]
            )

        with col2:
            internet_service = st.selectbox(
                "Internet Service", ["Yes", "No"]
            )
            internet_type = st.selectbox(
                "Internet Type",
                ["None", "DSL", "Fiber Optic", "Cable"]
            )
            avg_gb = st.number_input(
                "Average Monthly GB Download",
                min_value=0.0, value=10.0
            )

        with col3:
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "None"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "None"]
            )
            device_protection = st.selectbox(
                "Device Protection Plan", ["Yes", "No", "None"]
            )

        st.divider()
        st.subheader("🎬 Additional Services")

        col1, col2 = st.columns(2)

        with col1:
            tech_support = st.selectbox(
                "Premium Tech Support", ["Yes", "No", "None"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "None"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "None"]
            )
            streaming_music = st.selectbox(
                "Streaming Music", ["Yes", "No", "None"]
            )

        with col2:
            unlimited_data = st.selectbox(
                "Unlimited Data", ["Yes", "No", "None"]
            )
            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )
            payment_method = st.selectbox(
                "Payment Method",
                ["Bank Withdrawal", "Credit Card", "Mailed Check"]
            )

        st.divider()
        st.subheader("💳 Billing Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            monthly_charge = st.number_input(
                "Monthly Charge", min_value=0.0, value=70.0
            )
            total_charges = st.number_input(
                "Total Charges", min_value=0.0, value=840.0
            )

        with col2:
            total_refunds = st.number_input(
                "Total Refunds", min_value=0.0, value=0.0
            )
            extra_data = st.number_input(
                "Total Extra Data Charges", min_value=0.0, value=0.0
            )

        with col3:
            total_long_distance = st.number_input(
                "Total Long Distance Charges", min_value=0.0, value=0.0
            )

        st.divider()

        predict = st.form_submit_button(
            "🔍 Predict Churn",
            use_container_width=True,
            type="primary"
        )

    # --------------------------------
    # MAKE THE PREDICTION AFTER SUBMIT
    # --------------------------------

    if predict:

        raw_data = {
            "Gender": gender,
            "Age": age,
            "Married": married,
            "Number of Dependents": dependents,
            "Zip Code": zip_code,
            "Latitude": latitude,
            "Longitude": longitude,
            "Number of Referrals": referrals,
            "Tenure in Months": tenure,
            "Offer": offer,
            "Phone Service": phone_service,
            "Avg Monthly Long Distance Charges": long_distance,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_service,
            "Internet Type": internet_type,
            "Avg Monthly GB Download": avg_gb,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection Plan": device_protection,
            "Premium Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Streaming Music": streaming_music,
            "Unlimited Data": unlimited_data,
            "Contract": contract,
            "Paperless Billing": paperless,
            "Payment Method": payment_method,
            "Monthly Charge": monthly_charge,
            "Total Charges": total_charges,
            "Total Refunds": total_refunds,
            "Total Extra Data Charges": extra_data,
            "Total Long Distance Charges": total_long_distance
        }

        # Convert customer details into a DataFrame
        input_df = pd.DataFrame([raw_data])

        # Encode categorical features
        input_df = pd.get_dummies(
            input_df,
            drop_first=True,
            dtype=int
        )

        # Match the saved model's expected feature columns
        input_df = input_df.reindex(
            columns=encode_columns,
            fill_value=0
        )

        # Check that the input matches the model
        if input_df.shape[1] != model.n_features_in_:

            st.error(
                "The input features do not match the model's expected "
                "number of features. Please verify encode_columns."
            )

        else:
            # Predict the class
            prediction = model.predict(input_df)[0]

            # Find churn probability if available
            churn_probability = None

            if hasattr(model, "predict_proba"):
                classes = list(model.classes_)

                if 1 in classes:
                    churn_index = classes.index(1)
                    churn_probability = model.predict_proba(
                        input_df
                    )[0][churn_index]

            st.divider()
            st.subheader("📊 Prediction Result")

            result_col, probability_col = st.columns(2)

            with result_col:
                if prediction == 1:
                    st.error("⚠️ Model predicts: Churned")
                    st.write(
                        "This customer is predicted to leave."
                    )
                else:
                    st.success("✅ Model predicts: Stayed")
                    st.write(
                        "This customer is predicted to stay."
                    )

            with probability_col:
                if churn_probability is not None:
                    st.metric(
                        "Predicted Churn Probability",
                        f"{churn_probability:.1%}"
                    )

                    st.progress(float(churn_probability))

                    st.caption(
                        "This is the model's estimated probability, "
                        "not a guarantee of what the customer will do."
                    )
                else:
                    st.info(
                        "This model does not provide probability estimates."
                    )


# 10. Model information page
elif page == "About Model":

    st.title("About the ML Model")
    st.caption("Information about the model behind this application.")
    st.divider()

    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Task:** Binary classification")
    st.write("**Target:** Customer Status")
    st.write("**Classes:** Churned and Stayed")

    st.warning(
        "We will display actual test-set metrics here after verifying "
        "the evaluation results. We will not use made-up accuracy values."
    )

    st.subheader("Dataset details")
    st.write(f"Rows: {df.shape[0]:,}")
    st.write(f"Columns: {df.shape[1]:,}")
