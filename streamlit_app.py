import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Aspergillus niger Enzyme Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0px;
        color: #06b6d4;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 24px;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 16px;
    }
    .metric-number {
        font-size: 2.8rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-unit {
        font-size: 1.1rem;
        color: #06b6d4;
    }
</style>
""", unsafe_allow_html=True)

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    path = "model_artifacts.joblib"
    if not os.path.exists(path):
        st.error(f"Missing {path}. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(path)

artifacts = load_artifacts()
model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
valid_strains = [str(s) for s in artifacts["strains"]]
valid_substrates = artifacts["substrates"]

# Presets Definition
PRESETS = {
    "High-Yield Citrus Induction": {
        "strain": "4", "ph": 5.5, "substrate": "orange peel", "time_days": 5, "temperature": 28.0
    },
    "Extreme Hyper-Production Peak": {
        "strain": "2", "ph": 6.5, "substrate": "orange peel", "time_days": 4, "temperature": 50.0
    },
    "Wheat Bran Standard SSF": {
        "strain": "4", "ph": 5.0, "substrate": "wheat bran", "time_days": 5, "temperature": 28.0
    },
    "Rice Bran Lower Baseline": {
        "strain": "35", "ph": 6.5, "substrate": "rice bran", "time_days": 7, "temperature": 30.0
    },
    "Thermal Denaturation Stress": {
        "strain": "2", "ph": 6.5, "substrate": "rice bran", "time_days": 4, "temperature": 70.0
    }
}

# Session State for History & Preset Sync
if "history" not in st.session_state:
    st.session_state.history = []

if "current_preset" not in st.session_state:
    st.session_state.current_preset = "High-Yield Citrus Induction"

# Helper Prediction Function
def predict_activity(strain, ph, substrate, time_days, temperature):
    df_input = pd.DataFrame([{
        "strain": str(strain),
        "ph": float(ph),
        "substrate": str(substrate).strip().lower(),
        "time(days)": float(time_days),
        "temperature": float(temperature)
    }])
    X_trans = preprocessor.transform(df_input)
    mean_val, std_val = model.predict(X_trans, return_std=True)
    pred_mean = max(0.0, float(mean_val[0]))
    std_err = float(std_val[0])
    ci_lower = max(0.0, pred_mean - 1.96 * std_err)
    ci_upper = pred_mean + 1.96 * std_err
    return pred_mean, std_err, ci_lower, ci_upper

# Sidebar Presets
st.sidebar.title("🔬 Bioprocess Presets")
st.sidebar.markdown("Quickly load validated fermentation benchmarks:")

for name, params in PRESETS.items():
    if st.sidebar.button(name, use_container_width=True):
        st.session_state.selected_strain = params["strain"]
        st.session_state.selected_substrate = params["substrate"]
        st.session_state.selected_ph = params["ph"]
        st.session_state.selected_time = params["time_days"]
        st.session_state.selected_temp = params["temperature"]

st.sidebar.markdown("---")
st.sidebar.info("**Model Architecture:**\nGaussian Process Regressor with Radial Basis Function (RBF) and White Noise Kernels ($R^2 = +0.299$, $MAE = 140.90$ U/mL).")

# Main Header
st.markdown('<div class="main-title">Aspergillus niger Enzyme Activity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Gaussian Process Machine Learning • 95% Bayesian Confidence Intervals • Bioprocess Optimization</div>', unsafe_allow_html=True)

# Layout Columns
col_input, col_output = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("1. Inoculation & Growth Parameters")
    
    # Strain Selection
    default_strain = st.session_state.get("selected_strain", "4")
    strain_idx = valid_strains.index(default_strain) if default_strain in valid_strains else 0
    strain = st.selectbox("Fungal Strain Lineage", valid_strains, index=strain_idx, help="Isolate identifier of Aspergillus niger")

    # Substrate Selection
    default_sub = st.session_state.get("selected_substrate", "orange peel")
    sub_idx = valid_substrates.index(default_sub) if default_sub in valid_substrates else 0
    substrate = st.selectbox("Carbon / Agro-Biomass Substrate", valid_substrates, index=sub_idx, help="Solid substrate carbon & inducer source")

    # pH Slider
    default_ph = st.session_state.get("selected_ph", 5.5)
    ph = st.slider("Medium pH", min_value=4.0, max_value=7.0, value=float(default_ph), step=0.1, help="Optimal secretion occurs between pH 4.5 and 5.5")

    # Time Slider
    default_time = st.session_state.get("selected_time", 5)
    time_days = st.slider("Fermentation Duration (days)", min_value=4, max_value=7, value=int(default_time), step=1)

    # Temperature Slider
    default_temp = st.session_state.get("selected_temp", 28.0)
    temperature = st.slider("Cultivation Temperature (°C)", min_value=20.0, max_value=75.0, value=float(default_temp), step=0.5, help="Temperatures >= 65°C trigger irreversible thermal denaturation")

    predict_btn = st.button("⚡ Run Predictive Inference", type="primary", use_container_width=True)

# Run Prediction
pred_val, std_err, ci_lower, ci_upper = predict_activity(strain, ph, substrate, time_days, temperature)

# Check Thermal Alert
thermal_alert = temperature >= 65.0

# Add to History on button click
if predict_btn or len(st.session_state.history) == 0:
    st.session_state.history.insert(0, {
        "Timestamp": time.strftime("%H:%M:%S"),
        "Strain": strain,
        "Substrate": substrate,
        "pH": ph,
        "Duration": f"{time_days}d",
        "Temperature": f"{temperature:.1f}°C",
        "Predicted (U/mL)": round(pred_val, 2),
        "95% CI": f"[{ci_lower:.1f} - {ci_upper:.1f}]",
        "Status": "Thermal Denaturation" if thermal_alert else ("Strong Induction" if substrate in ["orange peel", "pectin"] and temperature <= 50 else "Normal")
    })
    if len(st.session_state.history) > 20:
        st.session_state.history.pop()

with col_output:
    st.subheader("2. Predictive Inference & Analysis")

    # Hero Metric Card
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Predicted Enzyme Activity</div>
        <div class="metric-number">{pred_val:.2f} <span class="metric-unit">U/mL</span></div>
        <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px;">
            95% Credible Interval: <strong>[{ci_lower:.1f} – {ci_upper:.1f} U/mL]</strong> (±{std_err:.1f} U/mL)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Biological Status Callout
    if thermal_alert:
        st.error("⚠️ **Thermal Denaturation Alert (≥65°C)**: Operating temperature exceeds fungal cell viability and enzyme tertiary folding stability. Active enzyme yield is degraded.")
    elif substrate in ["orange peel", "pectin"] and (4.5 <= ph <= 6.0) and (25 <= temperature <= 50):
        st.success("🚀 **Optimal Pectinolytic Induction**: Pectinaceous galacturonides induce intense GaaR-mediated transcription of fungal pectinase cascades.")
    elif substrate == "rice bran":
        st.warning("📊 **Subdued Secretion Profile**: Rice bran provides lower pectin concentration and higher starch/lipid ratios, resulting in lower enzyme production.")
    else:
        st.info("🔬 **Normal Physiological Production**: Fermentation conditions are within viable vegetative growth limits.")

    # Temperature Response Curve Plot
    st.markdown("#### Temperature Sensitivity Profile")
    test_temps = np.linspace(20, 75, 40)
    curve_preds = []
    curve_lowers = []
    curve_uppers = []

    for t in test_temps:
        m, s, lo, hi = predict_activity(strain, ph, substrate, time_days, t)
        curve_preds.append(m)
        curve_lowers.append(lo)
        curve_uppers.append(hi)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.fill_between(test_temps, curve_lowers, curve_uppers, color="#06b6d4", alpha=0.18, label="95% Confidence Band")
    ax.plot(test_temps, curve_preds, color="#06b6d4", linewidth=2.5, label="Mean Response")
    ax.axvspan(65, 75, color="#ef4444", alpha=0.12, label="Denaturation (≥65°C)")
    ax.axvline(temperature, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Current ({temperature}°C)")
    ax.scatter([temperature], [pred_val], color="#ef4444", s=60, zorder=5)

    ax.set_xlabel("Cultivation Temperature (°C)", fontsize=9)
    ax.set_ylabel("Activity (U/mL)", fontsize=9)
    ax.set_xlim(20, 75)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=7.5, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)

# History Section
st.markdown("---")
st.subheader("📋 Experiment Simulation Log")

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.table(hist_df)

    csv_data = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Simulation History (CSV)",
        data=csv_data,
        file_name="aspergillus_niger_simulations.csv",
        mime="text/csv"
    )
else:
    st.write("No predictions run yet.")
