import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Data ---
data = pd.DataFrame({
    "place": ["Lonavala", "Lonavala", "Lonavala", "Karjat", "Karjat", "Khandala", "Khandala", "Khandala"],
    "area":  [2600, 3000, 3200, 2600, 2800, 2600, 2900, 3200],
    "price": [5.5, 5.65, 6.1, 4.5, 5.0, 6.0, 6.5, 6.8],
})

VALID_PLACES = sorted(data["place"].unique())

# --- Model ---
preprocessor = ColumnTransformer(transformers=[
    ("place_enc", OneHotEncoder(handle_unknown="ignore"), ["place"]),
    ("area_pass", "passthrough", ["area"]),
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])

X = data[["place", "area"]]
y = data["price"]
model.fit(X, y)

# --- R² score ---
r2 = model.score(X, y)


def predict(place: str, area: float) -> float:
    """Return predicted price in Lakhs for given place and area."""
    input_df = pd.DataFrame({"place": [place], "area": [float(area)]})
    return round(float(model.predict(input_df)[0]), 2)


def main():
    print("=" * 45)
    print("       Flat Price Prediction Model")
    print("=" * 45)
    print(f"  Model R² (training fit): {r2:.4f}")
    print(f"  Locations available: {', '.join(VALID_PLACES)}")
    print("=" * 45)

    while True:
        print("\nEnter details (or 'quit' to exit):")

        # Location input
        place_input = input(f"  Location {VALID_PLACES}: ").strip()
        if place_input.lower() == "quit":
            print("Goodbye!")
            break

        # Validate location
        matched = next(
            (p for p in VALID_PLACES if p.lower() == place_input.lower()), None
        )
        if not matched:
            print(f"  [!] Unknown location '{place_input}'. Choose from: {', '.join(VALID_PLACES)}")
            continue

        # Area input
        area_input = input("  Area (sq ft): ").strip()
        if area_input.lower() == "quit":
            print("Goodbye!")
            break

        try:
            area_val = float(area_input)
            if area_val <= 0:
                raise ValueError
        except ValueError:
            print("  [!] Please enter a valid positive number for area.")
            continue

        price = predict(matched, area_val)
        print(f"\n  [*] Predicted Price: Rs. {price} Lakhs")
        print(f"      ({matched}, {area_val:,.0f} sq ft)")


if __name__ == "__main__":
    main()
