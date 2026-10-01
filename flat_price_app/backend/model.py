"""
Flat Price Prediction — ML Model
Trains a Linear Regression model with One-Hot Encoded location feature.
Exposes predict() and model metadata for use by the API.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
_DATA = pd.DataFrame({
    "place": [
        "Lonavala", "Lonavala", "Lonavala",
        "Karjat",   "Karjat",
        "Khandala", "Khandala", "Khandala",
    ],
    "area": [2600, 3000, 3200, 2600, 2800, 2600, 2900, 3200],
    "price": [5.5, 5.65, 6.1, 4.5, 5.0, 6.0, 6.5, 6.8],
})

VALID_PLACES: list[str] = sorted(_DATA["place"].unique().tolist())
AREA_MIN: int = int(_DATA["area"].min())
AREA_MAX: int = int(_DATA["area"].max())

# ---------------------------------------------------------------------------
# Build & train pipeline
# ---------------------------------------------------------------------------
_preprocessor = ColumnTransformer(transformers=[
    ("place_enc", OneHotEncoder(handle_unknown="ignore"), ["place"]),
    ("area_pass", "passthrough", ["area"]),
])

_pipeline = Pipeline(steps=[
    ("preprocessor", _preprocessor),
    ("regressor",    LinearRegression()),
])

_X = _DATA[["place", "area"]]
_y = _DATA["price"]
_pipeline.fit(_X, _y)

R2_SCORE: float = round(float(_pipeline.score(_X, _y)), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict(place: str, area: float) -> float:
    """
    Return predicted flat price (in Lakhs) for the given location and area.

    Parameters
    ----------
    place : str   One of VALID_PLACES (case-insensitive).
    area  : float Flat area in square feet (positive number).

    Returns
    -------
    float  Predicted price rounded to 2 decimal places.

    Raises
    ------
    ValueError if place is not recognised.
    """
    matched = next(
        (p for p in VALID_PLACES if p.lower() == place.strip().lower()), None
    )
    if matched is None:
        raise ValueError(
            f"Unknown location '{place}'. Valid options: {', '.join(VALID_PLACES)}"
        )
    if area <= 0:
        raise ValueError("Area must be a positive number.")

    row = pd.DataFrame({"place": [matched], "area": [float(area)]})
    return round(float(_pipeline.predict(row)[0]), 2)


def model_info() -> dict:
    """Return metadata about the trained model."""
    return {
        "algorithm":    "Linear Regression",
        "encoding":     "One-Hot Encoding (location)",
        "r2_score":     R2_SCORE,
        "valid_places": VALID_PLACES,
        "area_min":     AREA_MIN,
        "area_max":     AREA_MAX,
        "training_rows": len(_DATA),
    }
