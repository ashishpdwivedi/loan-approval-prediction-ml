
import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing objects
bundle = joblib.load("loan_approval_model.pkl")

model = bundle["model"]
preprocessor = bundle["preprocessor"]
columns = bundle["columns"]
category_options = bundle["category_options"]

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Predictor")
st.write(
    "Enter the applicant details below to get an "
    "ML-based loan prediction."
)

st.info(
    "This tool provides an estimate based on a trained "
    "machine learning model, not an actual lender decision."
)

with st.form("loan_form"):

    st.subheader("Applicant Information")

    person_age = st.number_input(
        "Applicant Age",
        min_value=18,
        max_value=100,
        value=28
    )

    person_gender = st.selectbox(
        "Gender",
        category_options["person_gender"]
    )

    person_education = st.selectbox(
        "Education",
        category_options["person_education"]
    )

    person_income = st.number_input(
        "Annual Income (dataset units)",
        min_value=1.0,
        value=80000.0,
        step=1000.0
    )

    person_emp_exp = st.number_input(
        "Employment Experience (years)",
        min_value=0,
        max_value=80,
        value=5
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        category_options["person_home_ownership"]
    )

    st.subheader("Loan Information")

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=500.0,
        value=10000.0,
        step=500.0
    )

    loan_intent = st.selectbox(
        "Loan Purpose",
        category_options["loan_intent"]
    )

    loan_int_rate = st.number_input(
        "Loan Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=11.0,
        step=0.1
    )

    loan_percent_income = st.number_input(
        "Loan Amount / Annual Income",
        min_value=0.0,
        max_value=1.0,
        value=0.125,
        step=0.01,
        help="Enter a ratio: for example, 0.20 means 20%."
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (years)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    previous_loan_defaults_on_file = st.selectbox(
        "Previous Loan Defaults",
        category_options["previous_loan_defaults_on_file"]
    )

    submitted = st.form_submit_button(
        "Check Loan Status",
        use_container_width=True
    )

if submitted:

    # Collect input using original training column names
    input_data = {
        "person_age": float(person_age),
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": float(person_income),
        "person_emp_exp": int(person_emp_exp),
        "person_home_ownership": person_home_ownership,
        "loan_amnt": float(loan_amnt),
        "loan_intent": loan_intent,
        "loan_int_rate": float(loan_int_rate),
        "loan_percent_income": float(loan_percent_income),
        "cb_person_cred_hist_length": float(
            cb_person_cred_hist_length
        ),
        "credit_score": int(credit_score),
        "previous_loan_defaults_on_file":
            previous_loan_defaults_on_file
    }

    input_df = pd.DataFrame(
        [input_data],
        columns=columns
    )

    # Validate derived ratio against the entered income/loan
    actual_ratio = loan_amnt / person_income

    if abs(actual_ratio - loan_percent_income) > 0.02:
        st.warning(
            "The entered loan-to-income ratio differs "
            "from loan amount divided by annual income. "
            "Please verify the ratio before proceeding."
        )

    # Transform raw inputs using the fitted encoder
    encoded_input = preprocessor.transform(input_df)

    # Predict class and probability
    prediction = int(model.predict(encoded_input)[0])
    probabilities = model.predict_proba(encoded_input)[0]

    # Find probability associated with class 1
    class1_index = list(model.classes_).index(1)
    class1_probability = probabilities[class1_index]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "🎉 Congratulations! Based on our model, "
            "your loan is predicted to be approved."
        )
    else:
        st.error(
            "Sorry! Based on our model, your loan is "
            "predicted not to be approved."
        )

    st.caption(
        "This is a model prediction, not a confirmed "
        "loan approval or rejection."
    )

    st.metric(
        "Model-estimated probability of Class 1",
        f"{class1_probability:.1%}"
    )
