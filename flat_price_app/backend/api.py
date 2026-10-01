"""
Flat Price Prediction — FastAPI Backend
Run with:  python -m uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import sys, os

# Allow imports from this folder when run directly
sys.path.insert(0, os.path.dirname(__file__))
from model import predict, model_info, VALID_PLACES

app = FastAPI(
    title="Flat Price Predictor API",
    description="Predict flat prices for Lonavala, Karjat, and Khandala.",
    version="1.0.0",
)

# Allow Streamlit frontend (running on any localhost port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    place: str
    area: float

    @field_validator("place")
    @classmethod
    def place_must_be_valid(cls, v: str) -> str:
        matched = next(
            (p for p in VALID_PLACES if p.lower() == v.strip().lower()), None
        )
        if matched is None:
            raise ValueError(
                f"Unknown location '{v}'. Valid: {', '.join(VALID_PLACES)}"
            )
        return matched

    @field_validator("area")
    @classmethod
    def area_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Area must be a positive number.")
        return v


class PredictResponse(BaseModel):
    place:           str
    area:            float
    predicted_price: float
    unit:            str = "Lakhs (Rs.)"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Flat Price Predictor API is running. Visit /docs for Swagger UI."}


@app.get("/info")
def info():
    """Return model metadata."""
    return model_info()


@app.post("/predict", response_model=PredictResponse)
def predict_price(req: PredictRequest):
    """Predict flat price given location and area."""
    try:
        price = predict(req.place, req.area)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictResponse(
        place=req.place,
        area=req.area,
        predicted_price=price,
    )
