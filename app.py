import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RC Beam Retrofit Predictor", layout="centered")

st.title("RC Beam Retrofit Predictor")
st.caption("ExtraTrees multi-output model — predicts capacity, deflections, and debonding onset")

@st.cache_resource
def load_artifacts():
    model = joblib.load("final_extratrees_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("top_features.json") as f:
        top_features = json.load(f)
    return model, scaler, top_features

try:
    model, scaler, top_features = load_artifacts()
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

output_names = [
    "Ultimate Load",
    "Ultimate Deflection",
    "First Yield Load",
    "First Yield Deflection",
    "Debonding Start Load",
    "Debonding Start Deflection",
]

with st.form("predict_form"):
    st.subheader("Input features")
    cols = st.columns(2)
    input_data = {}
    for i, feature in enumerate(top_features):
        with cols[i % 2]:
            input_data[feature] = st.number_input(feature, value=0.0, format="%.5f")
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([input_data])[top_features]
    try:
        input_scaled = scaler.transform(input_df)
        preds = model.predict(input_scaled)
        preds = np.asarray(preds).reshape(1, -1)  # ensure 2D
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Prediction Results")
    result_dict = {}
    for i, val in enumerate(preds[0]):
        label = output_names[i] if i < len(output_names) else f"Output {i+1}"
        result_dict[label] = float(val)

    res_df = pd.DataFrame([result_dict])
    st.dataframe(res_df, use_container_width=True)

    csv = pd.concat([input_df, res_df], axis=1).to_csv(index=False).encode("utf-8")
    st.download_button("Download Results (CSV)", data=csv, file_name="prediction_results.csv", mime="text/csv")

with st.expander("About this app"):
    st.markdown(
        """
        - Model: ExtraTreesRegressor (multi-output) trained on 291 FE simulations of epoxy-bonded steel plate retrofitted RC beams.
        - Interface modelling: Cohesive Zone Modeling (CZM).
        - Inputs must match the training feature scaling and order in `top_features.json`.
        - This tool is for research; verify outputs before design use.
        """
    )
