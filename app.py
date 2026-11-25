
import streamlit as st
import pandas as pd
import joblib
import base64
from pathlib import Path

# Load trained models and feature columns
lr_model = joblib.load("linear_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# Load background image
image_path = Path(__file__).parent / "house_background.png"

with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(
        image_file.read()
    ).decode()

# Apply background styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(
                rgba(15, 23, 42, 0.72),
                rgba(15, 23, 42, 0.82)
            ),
            url("data:image/png;base64,{encoded_image}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    h1, h2, h3, p, label {{
        color: white;
    }}

    [data-testid="stVerticalBlock"] {{
        color: white;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# Title
st.title("🏠 House Price Prediction")
st.write(
    "Predict house prices using Machine Learning. "
    "Enter the property details below to get a prediction."
)

st.divider()

# Model selection
st.subheader("1. Select Prediction Model")

model_choice = st.selectbox(
    "Choose your machine learning model",
    ["Linear Regression", "Random Forest"]
)

st.divider()

# Property details
st.subheader("2. Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (square feet)",
        min_value=100,
        max_value=50000,
        value=5000,
        step=100
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    stories = st.number_input(
        "Number of Stories",
        min_value=1,
        max_value=4,
        value=2
    )

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=5,
        value=1
    )

with col2:
    mainroad = st.selectbox("Main Road Access", ["yes", "no"])

    guestroom = st.selectbox("Guest Room", ["yes", "no"])

    basement = st.selectbox("Basement", ["yes", "no"])

    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        ["yes", "no"]
    )

    airconditioning = st.selectbox(
        "Air Conditioning",
        ["yes", "no"]
    )

    prefarea = st.selectbox(
        "Preferred Area",
        ["yes", "no"]
    )

    furnishingstatus = st.selectbox(
        "Furnishing Status",
        ["furnished", "semi-furnished", "unfurnished"]
    )

st.divider()

# Prediction
if st.button("Predict House Price", type="primary", use_container_width=True):

    # Prepare input data
    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,

        "mainroad_yes": mainroad == "yes",
        "guestroom_yes": guestroom == "yes",
        "basement_yes": basement == "yes",
        "hotwaterheating_yes": hotwaterheating == "yes",
        "airconditioning_yes": airconditioning == "yes",
        "prefarea_yes": prefarea == "yes",

        "furnishingstatus_semi-furnished":
            furnishingstatus == "semi-furnished",

        "furnishingstatus_unfurnished":
            furnishingstatus == "unfurnished"
    }

    # Arrange columns exactly as during model training
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Select model
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_df)[0]
    else:
        prediction = rf_model.predict(input_df)[0]

    # Display result
    st.success("Prediction completed successfully!")

    st.subheader("Predicted House Price")

    st.metric(
        label=f"Using {model_choice}",
        value=f"{prediction:,.2f}"
    )

    st.caption(
        "Prediction is an estimate based on the selected "
        "machine learning model and the dataset's price units."
    )

st.divider()

st.caption("House Price Prediction | Machine Learning Project")
