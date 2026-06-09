###############################################
# app.py — FINAL (Dynamic region features)
###############################################

import streamlit as st
import yaml
import os
import joblib
import pandas as pd
from streamlit_folium import st_folium

from src.geo_utils import folium_risk_areas
from src.recommendations import generate_recommendations

st.set_page_config(layout="wide", page_title="Disaster Management Prototype")

# ---------------------------------------
# LOAD CONFIG
# ---------------------------------------
cfg = yaml.safe_load(open("config.yaml"))
regions = list(cfg["regions"].keys())

st.title("🌍 Disaster Risk Prediction System")

# ---------------------------------------
# SIDEBAR
# ---------------------------------------
region_key = st.sidebar.selectbox("Select Region", regions)
region = cfg["regions"][region_key]

st.sidebar.write(f"**Region:** {region['name']}")
st.sidebar.write(f"**Hazard:** {region['problem']}")

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    dataset_path = os.path.join(
        region["data_path"], "raw", f"{region_key}_tabular.csv"
    )
    if not os.path.exists(dataset_path):
        st.error("Dataset not found")
        st.stop()
    df = pd.read_csv(dataset_path)

st.subheader("📄 Data Preview")
st.dataframe(df.head())

# ---------------------------------------
# LOAD MODEL BUNDLE
# ---------------------------------------
bundle_path = f"models/{region_key}_best_model.joblib"

if not os.path.exists(bundle_path):
    st.error("Model not trained. Run train.py first.")
    st.stop()

bundle = joblib.load(bundle_path)

imputer = bundle["imputer"]
scaler = bundle["scaler"]
model = bundle["model"]
feature_cols = bundle["feature_cols"]
best_model_name = bundle["best_model_name"]

st.info(f"Best model selected: **{best_model_name}**")

# ---------------------------------------
# PREPARE FEATURES
# ---------------------------------------
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

X = df[feature_cols]
X = imputer.transform(X)
X = scaler.transform(X)

# ---------------------------------------
# PREDICTION
# ---------------------------------------
st.subheader("🔮 Predict Risk")

if st.button("Predict"):

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df_out = df.copy()
    df_out["pred_prob"] = probs
    df_out["pred_label"] = preds

    st.dataframe(df_out.head())

    out_csv = f"models/predictions_{region_key}.csv"
    df_out.to_csv(out_csv, index=False)
    st.success(f"Predictions saved → {out_csv}")

    # ---------------------------------------
    # RECOMMENDATIONS
    # ---------------------------------------
    region_meta = {"name": region["name"], "problem": region["problem"]}
    rec = generate_recommendations(df_out, region_meta=region_meta)

    st.subheader("🚨 AI Recommendations")
    st.write(f"Severity: **{rec['severity']}**")
    st.write(f"Mean risk: **{rec['mean_prob']:.2f}**")

    for m in rec["messages"]:
        st.write("- " + m)

    for a in rec["suggested_actions"]:
        st.write("- " + a)

    # ---------------------------------------
    # MAP
    # ---------------------------------------
    st.subheader("🗺 Risk Map")

    m = folium_risk_areas(
        df_out,
        lat_col="lat",
        lon_col="lon",
        prob_col="pred_prob",
        prob_threshold=0.6,
        buffer_deg=2 / 111.0,
        center=region["coordinates"],
        zoom_start=11
    )

    st_folium(m, width=900, height=650)
    
    st.stop()
