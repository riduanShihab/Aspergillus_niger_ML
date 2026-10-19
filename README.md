# 🔬 *Aspergillus niger* Enzyme Activity Machine Learning Predictor

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Model-Gaussian_Process_Regression-success)]()

A comprehensive computational biology and machine learning system for modeling, optimizing, and predicting **fungal enzyme activity (U/mL)** in ***Aspergillus niger*** solid-state and submerged fermentations.

This repository features an audited experimental dataset, a benchmarked 5×5 Repeated K-Fold cross-validation workflow, and **two deployment-ready interactive interfaces** (a custom **HTML5/FastAPI web application** and a **Streamlit dashboard**), complete with **95% Bayesian Credible Intervals ($\pm 1.96\sigma$)** and bioprocess hazard detection.

---

## 📌 Table of Contents
- [Biological Context & Research Background](#-biological-context--research-background)
- [Machine Learning Benchmark](#-machine-learning-benchmark)
- [Key Biochemical Insights](#-key-biochemical-insights)
- [Two Interactive UI Options](#-two-interactive-ui-options)
  - [Option A: Modern HTML5 / FastAPI Web App](#option-a-modern-html5--fastapi-web-application-recommended)
  - [Option B: Streamlit Dashboard](#option-b-streamlit-dashboard)
- [REST API Reference](#-rest-api-reference)
- [Jupyter Research Notebook](#-jupyter-research-notebook)
- [Repository Structure](#-repository-structure)
- [Installation & Quick Start](#-installation--quick-start)

---

## 🧬 Biological Context & Research Background

Filamentous fungi such as ***Aspergillus niger*** are primary industrial workhorses for secreting hydrolytic enzymes (pectinases, cellulases, xylanases, amylases, and proteases). The yield of secreted enzymes is governed by multi-variable non-linear biological interactions:

1. **Fungal Strain Lineage (`strain`)**: Tested isolate variants (`Strain 2`, `Strain 4`, `Strain 12`, `Strain 35`).
2. **Agro-Biomass Substrate (`substrate`)**: Solid-state inducers including **Orange Peel**, **Pectin**, **Banana Peel**, **Wheat Bran**, **Rice Bran**, and **Potato Peel**.
3. **Medium pH (`pH`)**: Controls cell membrane proton-motive gradients, nutrient permeases, and catalytic active site ionization ($4.0 - 7.0$).
4. **Fermentation Duration (`time(days)`)**: Governs fungal vegetative vs. idiophase/production kinetics ($4 - 7$ days).
5. **Cultivation Temperature (`temperature`)**: Regulates metabolic rates and tertiary protein folding stability ($20^\circ\text{C} - 75^\circ\text{C}$).

---

## 📊 Machine Learning Benchmark

The dataset was benchmarked using a rigorous **5-Fold Cross-Validation with 5 Repeats (25 total validation iterations)** to guard against small-sample variance ($N=34$ clean laboratory runs).

| Rank | Model Architecture | MAE (U/mL) | RMSE (U/mL) | $R^2$ Score | Key Characteristics |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **Gaussian Process Regressor (GPR)** | **140.90** | **197.53** | **+0.299** | **Top Performer**: Non-parametric Bayesian model with RBF + White noise kernels; outputs exact $95\%$ credible bounds. |
| 🥈 | **Random Forest Regressor** | 156.67 | 209.41 | +0.201 | Captures non-linear thresholds and thermal cutoff step-functions. |
| 🥉 | **Ridge Regression (Target-Scaled)** | 159.05 | 226.36 | -0.099 | Linear baseline with target standard scaling. |
| 4 | **Support Vector Regression (SVR)** | 160.04 | 217.64 | +0.155 | **Fixed**: Target scaling via `TransformedTargetRegressor` repaired default RBF collapse (previously $R^2 = -0.353$). |
| 5 | **Gradient Boosting Regressor** | 166.76 | 224.80 | +0.011 | Shallow tree boosting baseline. |
| 6 | **Baseline (Dummy Mean)** | 207.03 | 259.15 | -0.243 | Reference naive mean predictor. |

### Feature Importance (Random Forest MDI & GPR Sensitivity):
1. **Substrate: Rice Bran (~28.0%)**: Strongest negative predictor of high enzyme titer.
2. **Cultivation Temperature (~23.6%)**: Critical non-linear parameter dictating active vs. denatured states.
3. **Medium pH (~17.0%)**: High sensitivity with an optimal peak between pH $5.0$ and $5.5$.
4. **Substrate: Orange Peel (~16.9%)**: Potent biochemical inducer triggering hyper-secretion.
5. **Fermentation Time & Strain Variant (< 5% each)**: Secondary modulation parameters.

---

## 💡 Key Biochemical Insights
* **Pectinolytic Cascade Induction**: Orange peel and pure pectin yield peak activities (up to **$1499.4\text{ U/mL}$** in laboratory trials) via *GaaR*-mediated transcription factor activation.
* **Thermal Denaturation Cutoff ($\ge 65^\circ\text{C}$)**: Irreversible protein unfolding triggers an immediate drop to **$0.0\text{ U/mL}$** activity at $70^\circ\text{C}$. Both UIs include automated biological hazard alerts when temperature exceeds $65^\circ\text{C}$.

---

## 🖥️ Two Interactive UI Options

You have **two complete UI frontends** available in this repository:

### Option A: Modern HTML5 / FastAPI Web Application *(Recommended)*
A high-performance, dark-mode, glassmorphic biotech web interface with zero full-page reloads and continuous real-time responsiveness.

* **Start the App**:
  ```bash
  python app.py
  ```
* **Access URL**: Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.
* **Features**:
  * **Dynamic Canvas Temperature Curve**: Smoothly renders the temperature sensitivity profile ($20^\circ\text{C} - 75^\circ\text{C}$) with shaded 95% Bayesian confidence intervals and a live marker for your current input.
  * **Interactive Substrate Tiles**: Visual selection with emoji icons and real-time active glow.
  * **1-Click Presets**: Fast-loading benchmark scenarios (Citrus Optimal, Extreme Peak, Wheat Bran Standard, Rice Bran Baseline, Thermal Stress).
  * **Experiment Simulation Log**: Keeps a running record of all simulation runs with a 1-click **Export CSV** button.

---

### Option B: Streamlit Dashboard
A clean, Python-native dashboard ideal for rapid data exploration and 1-click deployment to [Streamlit Community Cloud](https://share.streamlit.io).

* **Start the App**:
  ```bash
  python -m streamlit run streamlit_app.py --server.port 8501
  ```
* **Access URL**: Open **[http://localhost:8501](http://localhost:8501)** in your browser.
* **Features**:
  * Sidebar presets for 1-click benchmark configuration.
  * Interactive parameter sliders with real-time prediction cards.
  * Matplotlib sensitivity curve with 95% confidence bands and denaturation warning zones.
  * Tabular experiment log with CSV export.

---

## 🌐 REST API Reference

When running `python app.py`, the backend exposes an OpenAPI-compliant REST API:

### `POST /api/predict`
Calculates predicted enzyme activity, credible bounds, biological status, and a full 25-point temperature sensitivity curve.

**Request Body (`application/json`):**
```json
{
  "strain": "4",
  "ph": 5.5,
  "substrate": "orange peel",
  "time_days": 5,
  "temperature": 28.0
}
```

**Response Body (`application/json`):**
```json
{
  "predicted_activity": 594.52,
  "std_error": 180.14,
  "ci_95_lower": 241.44,
  "ci_95_upper": 947.60,
  "status": "Strong Pectinolytic Induction",
  "status_color": "#10b981",
  "thermal_alert": false,
  "bioprocess_notes": "High concentrations of pectinaceous galacturonides induce intense GaaR-mediated transcriptional activation...",
  "temperature_curve": [
    {"temperature": 20.0, "predicted_activity": 564.12, "ci_lower": 208.30, "ci_upper": 919.94},
    ...
  ]
}
```

### Other Endpoints:
* `GET /api/presets`: Retrieves validated laboratory benchmark parameter sets.
* `GET /api/metadata`: Retrieves valid strain IDs, substrate lists, and parameter ranges.
* `GET /docs`: Interactive Swagger API documentation.

---

## 📓 Jupyter Research Notebook

The research notebook [`Aspergillus_niger_ML_Research.ipynb`](file:///e:/Aspergillus_niger_ML/Aspergillus_niger_ML_Research.ipynb) provides a fully documented, publication-ready research trajectory following ML best practices:
1. **Setup & Environment-Resilient Ingestion**: Automatic detection of local CSV files or Google Colab Google Drive mounts.
2. **Exploratory Data Analysis (EDA)**: 4-panel publication visualization suite.
3. **Machine Learning Pipeline**: Complete cross-validation benchmarking with scikit-learn pipelines.
4. **Model Diagnostics**: Out-of-fold Actual vs. Predicted scatter plot and feature importance ranking.
5. **Interactive Prediction Widget**: In-notebook `ipywidgets` dashboard with automated fallback test scenarios.

---

## 📁 Repository Structure

```
Aspergillus_niger_ML/
├── Aspergillus_niger_ML_Research.ipynb   # Comprehensive research notebook with pre-rendered figures
├── app.py                                # FastAPI backend & web server
├── streamlit_app.py                      # Streamlit interactive dashboard
├── train_model.py                        # Production model training & serialization script
├── model_artifacts.joblib                # Serialized Gaussian Process model & ColumnTransformer
├── asergillus niger final.csv            # Cleaned experimental dataset (34 observations)
├── asergillus niger final_raw.csv        # Backup copy of original raw CSV
├── static/                               # Web UI assets for FastAPI app
│   ├── index.html                        # Semantic HTML5 single-page application
│   ├── style.css                         # Dark-mode glassmorphic styling
│   └── app.js                            # Client-side controller & Canvas temperature curve
└── README.md                             # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Aspergillus_niger_ML.git
cd Aspergillus_niger_ML
```

### 2. Install Dependencies
```bash
pip install fastapi uvicorn streamlit scikit-learn pandas numpy matplotlib seaborn "pyarrow<18" joblib
```

### 3. Launch the Application of Your Choice

* **To run the HTML5/FastAPI Web Application:**
  ```bash
  python app.py
  ```
  *Visit [http://127.0.0.1:8000](http://127.0.0.1:8000)*

* **To run the Streamlit Dashboard:**
  ```bash
  python -m streamlit run streamlit_app.py --server.port 8501
  ```
  *Visit [http://localhost:8501](http://localhost:8501)*

* **To open the Jupyter Notebook:**
  ```bash
  jupyter notebook Aspergillus_niger_ML_Research.ipynb
  ```

---

## 📄 License
This project is licensed under the Apache 2.0 License.
