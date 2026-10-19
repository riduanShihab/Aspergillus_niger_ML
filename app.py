import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="Aspergillus niger Enzyme Predictor",
    description="Machine Learning Bioprocess Prediction API for Fungal Enzyme Activity",
    version="1.0.0"
)

# Load trained production artifacts
ARTIFACTS_PATH = "model_artifacts.joblib"
if not os.path.exists(ARTIFACTS_PATH):
    raise FileNotFoundError(f"Artifacts file {ARTIFACTS_PATH} not found. Run train_model.py first.")

artifacts = joblib.load(ARTIFACTS_PATH)
model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
valid_strains = [str(s) for s in artifacts["strains"]]
valid_substrates = artifacts["substrates"]
ranges = artifacts["ranges"]

class PredictionRequest(BaseModel):
    strain: str = Field(default="4", description="Inoculated A. niger strain identifier")
    ph: float = Field(default=5.5, ge=3.5, le=8.0, description="Medium pH")
    substrate: str = Field(default="orange peel", description="Carbon / agro-biomass substrate")
    time_days: float = Field(default=5.0, ge=3.0, le=10.0, description="Fermentation duration in days")
    temperature: float = Field(default=30.0, ge=15.0, le=80.0, description="Incubation temperature in °C")

def compute_single_prediction(strain_val, ph_val, substrate_val, time_val, temp_val):
    df_input = pd.DataFrame([{
        "strain": str(strain_val),
        "ph": float(ph_val),
        "substrate": str(substrate_val).strip().lower(),
        "time(days)": float(time_val),
        "temperature": float(temp_val)
    }])
    X_trans = preprocessor.transform(df_input)
    mean_val, std_val = model.predict(X_trans, return_std=True)
    
    pred_mean = max(0.0, float(mean_val[0]))
    std_err = float(std_val[0])
    ci_lower = max(0.0, pred_mean - 1.96 * std_err)
    ci_upper = pred_mean + 1.96 * std_err
    return pred_mean, std_err, ci_lower, ci_upper

@app.get("/api/metadata")
def get_metadata():
    return {
        "strains": valid_strains,
        "substrates": valid_substrates,
        "ranges": ranges,
        "model_type": "Gaussian Process Regressor (Bayesian Optimization Ready)"
    }

@app.get("/api/presets")
def get_presets():
    return [
        {
            "id": "citrus-optimal",
            "name": "High-Yield Citrus Induction",
            "tag": "Optimal Yield",
            "params": {"strain": "4", "ph": 5.5, "substrate": "orange peel", "time_days": 5, "temperature": 28}
        },
        {
            "id": "hyper-peak",
            "name": "Extreme Hyper-Production Peak",
            "tag": "Max Record",
            "params": {"strain": "2", "ph": 6.5, "substrate": "orange peel", "time_days": 4, "temperature": 50}
        },
        {
            "id": "wheat-standard",
            "name": "Wheat Bran Standard SSF",
            "tag": "Reliable",
            "params": {"strain": "4", "ph": 5.0, "substrate": "wheat bran", "time_days": 5, "temperature": 28}
        },
        {
            "id": "rice-baseline",
            "name": "Rice Bran Lower Baseline",
            "tag": "Low Yield",
            "params": {"strain": "35", "ph": 6.5, "substrate": "rice bran", "time_days": 7, "temperature": 30}
        },
        {
            "id": "thermal-kill",
            "name": "Thermal Denaturation Stress",
            "tag": "Denatured",
            "params": {"strain": "2", "ph": 6.5, "substrate": "rice bran", "time_days": 4, "temperature": 70}
        }
    ]

@app.post("/api/predict")
def predict_endpoint(req: PredictionRequest):
    sub = req.substrate.strip().lower()
    if sub not in valid_substrates:
        # Check standard mapping
        if sub in ["wheat barn", "wheat bran"]:
            sub = "wheat bran"
        elif sub in ["rice barn", "rice bran"]:
            sub = "rice bran"
        else:
            raise HTTPException(status_code=400, detail=f"Invalid substrate. Must be one of: {valid_substrates}")
            
    st = str(req.strain).replace(".0", "")
    if st not in valid_strains:
        raise HTTPException(status_code=400, detail=f"Invalid strain. Must be one of: {valid_strains}")

    mean_pred, std_err, ci_lower, ci_upper = compute_single_prediction(
        st, req.ph, sub, req.time_days, req.temperature
    )

    # Biological evaluation & status
    thermal_alert = req.temperature >= 65.0
    if thermal_alert:
        status = "Thermal Denaturation"
        status_color = "#ef4444" # red
        bio_notes = "High temperature (≥65°C) exceeds cellular thermal viability and causes irreversible tertiary protein denaturation. Expect minimal or zero active enzyme yield."
    elif sub in ["orange peel", "pectin"] and (4.5 <= req.ph <= 6.0) and (25 <= req.temperature <= 50):
        status = "Strong Pectinolytic Induction"
        status_color = "#10b981" # green
        bio_notes = "High concentrations of pectinaceous galacturonides induce intense GaaR-mediated transcriptional activation of fungal pectinase cascades."
    elif sub == "rice bran":
        status = "Subdued Enzyme Secretion"
        status_color = "#f59e0b" # amber
        bio_notes = "Rice bran contains low pectin concentration and higher starch/lipid ratios, resulting in lower enzyme production compared to fruit peels."
    else:
        status = "Normal Physiological Production"
        status_color = "#3b82f6" # blue
        bio_notes = "Standard vegetative fermentation conditions within viable metabolic limits."

    # Compute temperature sensitivity curve for the chosen parameters
    temp_curve = []
    test_temps = np.linspace(20, 75, 25)
    for t in test_temps:
        t_val = round(float(t), 1)
        m, s, lo, hi = compute_single_prediction(st, req.ph, sub, req.time_days, t_val)
        temp_curve.append({
            "temperature": t_val,
            "predicted_activity": round(m, 2),
            "ci_lower": round(lo, 2),
            "ci_upper": round(hi, 2)
        })

    return {
        "predicted_activity": round(mean_pred, 2),
        "std_error": round(std_err, 2),
        "ci_95_lower": round(ci_lower, 2),
        "ci_95_upper": round(ci_upper, 2),
        "status": status,
        "status_color": status_color,
        "thermal_alert": thermal_alert,
        "bioprocess_notes": bio_notes,
        "temperature_curve": temp_curve
    }

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    print("Starting Aspergillus niger ML Prediction Server at http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
