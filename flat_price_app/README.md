# 🏠 Flat Price Predictor

A full-stack machine learning web application that predicts flat prices for **Lonavala**, **Karjat**, and **Khandala** based on location and area.

---

## 📌 Project Overview

| Detail | Value |
|---|---|
| **Model** | Linear Regression |
| **Encoding** | One-Hot Encoding (location) |
| **R² Score** | 0.9754 |
| **Backend** | FastAPI (REST API) |
| **Frontend** | Streamlit (Web UI) |
| **Dataset** | `flat_price.csv` (8 rows) |

---

## 📁 Project Structure

```
flat_price_app/
├── backend/
│   ├── model.py          # ML model — trains & exposes predict()
│   └── api.py            # FastAPI REST API
├── frontend/
│   └── app.py            # Streamlit web UI
├── requirements.txt      # Python dependencies
flat_price.csv            # Training dataset
flat_price_model.py       # Standalone CLI prediction script
```

---

## ⚙️ Setup & Installation

### 1. Clone or unzip the project

```bash
git clone https://github.com/YOUR_USERNAME/flat-price-predictor.git
cd flat-price-predictor
```

### 2. Install dependencies

```bash
pip install -r flat_price_app/requirements.txt
```

---

## 🚀 Running the App

You need **two terminals** — one for the backend, one for the frontend.

### Terminal 1 — Start the Backend API

```bash
cd flat_price_app/backend
python -m uvicorn api:app --reload --port 8000
```

API will be available at: **http://localhost:8000**
Swagger docs at: **http://localhost:8000/docs**

### Terminal 2 — Start the Frontend UI

```bash
cd flat_price_app/frontend
python -m streamlit run app.py --server.port 8501
```

App will open automatically at: **http://localhost:8501**

---

## 🔌 API Reference

### `GET /info`
Returns model metadata (algorithm, R² score, valid locations, area range).

### `POST /predict`
Predict flat price given location and area.

**Request body:**
```json
{
  "place": "Khandala",
  "area": 3000
}
```

**Response:**
```json
{
  "place": "Khandala",
  "area": 3000.0,
  "predicted_price": 6.55,
  "unit": "Lakhs (Rs.)"
}
```

---

## 🖥️ CLI Usage (Standalone)

Run predictions directly from the terminal without the web UI:

```bash
python flat_price_model.py
```

---

## 📊 Dataset

| Location | Area (sq ft) | Price (Lakhs) |
|---|---|---|
| Lonavala | 2600 | 5.50 |
| Lonavala | 3000 | 5.65 |
| Lonavala | 3200 | 6.10 |
| Karjat | 2600 | 4.50 |
| Karjat | 2800 | 5.00 |
| Khandala | 2600 | 6.00 |
| Khandala | 2900 | 6.50 |
| Khandala | 3200 | 6.80 |

---

## 📦 Dependencies

See [`requirements.txt`](flat_price_app/requirements.txt) for the full list.

| Package | Purpose |
|---|---|
| `scikit-learn` | Linear Regression + One-Hot Encoding |
| `pandas` / `numpy` | Data handling |
| `fastapi` + `uvicorn` | REST API backend |
| `streamlit` | Web frontend |
| `pydantic` | Request/response validation |
| `requests` | Frontend → backend HTTP calls |

---

## 👤 Author

Made with ❤️ using Python, FastAPI, and Streamlit.
