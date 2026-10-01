"""
Flat Price Prediction — Streamlit App (Streamlit Cloud compatible)
Model is embedded directly — no separate backend needed.
Run locally:  python -m streamlit run app.py
Deploy:       https://share.streamlit.io
"""

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flat Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Model (trained once, cached for the session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    data = pd.DataFrame({
        "place": [
            "Lonavala", "Lonavala", "Lonavala",
            "Karjat",   "Karjat",
            "Khandala", "Khandala", "Khandala",
        ],
        "area":  [2600, 3000, 3200, 2600, 2800, 2600, 2900, 3200],
        "price": [5.5,  5.65, 6.1,  4.5,  5.0,  6.0,  6.5,  6.8],
    })

    preprocessor = ColumnTransformer(transformers=[
        ("place_enc", OneHotEncoder(handle_unknown="ignore"), ["place"]),
        ("area_pass", "passthrough", ["area"]),
    ])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor",    LinearRegression()),
    ])
    pipeline.fit(data[["place", "area"]], data["price"])
    r2 = round(pipeline.score(data[["place", "area"]], data["price"]), 4)
    return pipeline, data, r2

pipeline, training_data, r2 = load_model()

VALID_PLACES = sorted(training_data["place"].unique().tolist())

def predict(place: str, area: float) -> float:
    row = pd.DataFrame({"place": [place], "area": [float(area)]})
    return round(float(pipeline.predict(row)[0]), 2)

# ---------------------------------------------------------------------------
# Sidebar — model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model Info")
    st.metric("Algorithm",     "Linear Regression")
    st.metric("R² Score",      r2)
    st.metric("Training Rows", len(training_data))
    st.markdown("---")
    st.caption("**Valid Locations**")
    for p in VALID_PLACES:
        st.markdown(f"- {p}")
    st.markdown("---")
    st.caption(
        f"Area range in training data:  \n"
        f"**{int(training_data['area'].min()):,} – {int(training_data['area'].max()):,} sq ft**"
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏠 Flat Price Predictor")
st.markdown(
    "Predict flat prices for **Lonavala**, **Karjat**, and **Khandala** "
    "using a Linear Regression model."
)
st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Enter Flat Details")

col1, col2 = st.columns(2)

with col1:
    place = st.selectbox(
        "Location",
        options=VALID_PLACES,
        help="Select the location of the flat.",
    )

with col2:
    area = st.number_input(
        "Area (sq ft)",
        min_value=500,
        max_value=10000,
        value=2800,
        step=100,
        help="Enter the flat area in square feet.",
    )

predict_btn = st.button("Predict Price", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------
if predict_btn:
    price = predict(place, area)

    st.divider()
    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Location",        place)
    c2.metric("Area",            f"{area:,} sq ft")
    c3.metric("Predicted Price", f"Rs. {price} L")

    st.success(
        f"Estimated flat price in **{place}** for **{area:,} sq ft** "
        f"is **Rs. {price} Lakhs**."
    )

# ---------------------------------------------------------------------------
# Training data reference
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Training Data Reference")

display_df = training_data.rename(columns={
    "place": "Location",
    "area":  "Area (sq ft)",
    "price": "Price (Lakhs)",
})
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Price trend charts per location
# ---------------------------------------------------------------------------
st.subheader("Price Trend by Location")
for loc in VALID_PLACES:
    loc_df = (
        training_data[training_data["place"] == loc][["area", "price"]]
        .set_index("area")
        .rename(columns={"price": f"{loc} — Price (Lakhs)"})
    )
    st.line_chart(loc_df, use_container_width=True, height=180)
    st.caption(f"Price trend — {loc}")
