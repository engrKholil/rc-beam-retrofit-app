# ============================================================
# PART 1
# IMPORTS + MODEL LOADING + PREDICTION ENGINE + SHAP
# ============================================================

import os
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Debonding Prediction System",
    page_icon="🏗️",
    layout="wide"
)


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR = "Final_Models"



# ============================================================
# LOAD ALL MODELS
# ============================================================

@st.cache_resource
def load_models():

    classifier = joblib.load(
        os.path.join(
            MODEL_DIR,
            "ET_Debonding_Classifier.pkl"
        )
    )


    global_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "Global_Response_Model.pkl"
        )
    )


    debond_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "Debonding_Response_Model.pkl"
        )
    )


    classifier_scaler = joblib.load(
        os.path.join(
            MODEL_DIR,
            "Classifier_Scaler.pkl"
        )
    )


    global_scaler = joblib.load(
        os.path.join(
            MODEL_DIR,
            "Global_Scaler.pkl"
        )
    )


    debond_scaler = joblib.load(
        os.path.join(
            MODEL_DIR,
            "Debonding_Scaler.pkl"
        )
    )


    classifier_features = joblib.load(
        os.path.join(
            MODEL_DIR,
            "Classifier_Features.pkl"
        )
    )


    global_features = joblib.load(
        os.path.join(
            MODEL_DIR,
            "Global_Features.pkl"
        )
    )


    debond_features = joblib.load(
        os.path.join(
            MODEL_DIR,
            "Debonding_Features.pkl"
        )
    )


    return (

        classifier,
        global_model,
        debond_model,

        classifier_scaler,
        global_scaler,
        debond_scaler,

        classifier_features,
        global_features,
        debond_features

    )



(
    classifier,
    global_model,
    debond_model,

    classifier_scaler,
    global_scaler,
    debond_scaler,

    classifier_features,
    global_features,
    debond_features

) = load_models()



# ============================================================
# SHAP EXPLAINER
# FIXED FOR STREAMLIT + SHAP
# ============================================================


@st.cache_resource
def create_shap_explainer():

    explainer = shap.TreeExplainer(
        classifier
    )

    return explainer



shap_explainer = create_shap_explainer()



# ============================================================
# FIXED PARAMETERS FOR SIMPLE MODE
# ============================================================

FIXED_PARAMETERS = {


    "Enn":144,
    "Ess":60,
    "σn":4,
    "Gc":0.05,

    "Ep":125000,
    "Er":165000,

    "fyp":300,
    "fyr":337,

    "fup":375,
    "fur":404.4,

    "εup":0.2,
    "εur":0.2

}



# ============================================================
# PREDICTION FUNCTION
# ============================================================


def predict_single(input_data):


    sample = pd.DataFrame(
        [input_data]
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    X_cls = sample[
        classifier_features
    ]


    X_cls_scaled = classifier_scaler.transform(
        X_cls
    )


    debond_flag = classifier.predict(
        X_cls_scaled
    )[0]


    probability = classifier.predict_proba(
        X_cls_scaled
    )[0,1]



    # --------------------------------------------------------
    # GLOBAL RESPONSE
    # --------------------------------------------------------

    X_global = sample[
        global_features
    ]


    X_global_scaled = global_scaler.transform(
        X_global
    )


    global_prediction = global_model.predict(
        X_global_scaled
    )[0]



    result = {


        "Debonding":
        "Yes" if debond_flag == 1 else "No",


        "Debonding Probability":
        round(float(probability),4),


        "Pu":
        round(float(global_prediction[0]),3),


        "δu":
        round(float(global_prediction[1]),3),


        "Py":
        round(float(global_prediction[2]),3),


        "δy":
        round(float(global_prediction[3]),3)

    }



    # --------------------------------------------------------
    # DEBONDING RESPONSE
    # --------------------------------------------------------

    if debond_flag == 1:


        X_deb = sample[
            debond_features
        ]


        X_deb_scaled = debond_scaler.transform(
            X_deb
        )


        deb_prediction = debond_model.predict(
            X_deb_scaled
        )[0]



        result["Pdeb"] = round(
            float(deb_prediction[0]),
            3
        )


        result["δdeb"] = round(
            float(deb_prediction[1]),
            3
        )


    else:

        result["Pdeb"] = None
        result["δdeb"] = None



    return result

# ============================================================
# PART 2
# STREAMLIT USER INTERFACE
# SIMPLE / ADVANCED MODE
# SINGLE PREDICTION
# ============================================================


# ============================================================
# TITLE
# ============================================================

st.title(
    "🏗️ AI-Based RC Beam Debonding Prediction System"
)


st.write(
"""
This AI system predicts:

✅ Debonding occurrence  
✅ Debonding probability  
✅ Ultimate load (Pu)  
✅ Ultimate displacement (δu)  
✅ Yield load (Py)  
✅ Yield displacement (δy)  
✅ Debonding load (Pdeb)  
✅ Debonding displacement (δdeb)

using machine learning models.
"""
)


st.divider()



# ============================================================
# TABS
# ============================================================

tab_prediction, tab_shap, tab_batch = st.tabs(

    [
        "🏗️ Prediction",
        "📊 SHAP Explainability",
        "📂 Batch Prediction"
    ]

)



# ============================================================
# PREDICTION TAB
# ============================================================

with tab_prediction:



    # --------------------------------------------------------
    # MODE SELECTION
    # --------------------------------------------------------

    mode = st.radio(

        "Select Input Mode",

        [
            "Simple Mode",
            "Advanced Mode"
        ],

        horizontal=True

    )



    input_values = {}



    # ========================================================
    # SIMPLE MODE
    # ========================================================

    if mode == "Simple Mode":


        st.subheader(
            "⚡ Simple Mode"
        )


        st.info(
"""
Only important structural parameters are required.

The remaining material/interface parameters are
automatically assigned from the reference model database.
"""
        )



        col1, col2, col3 = st.columns(3)



        with col1:


            input_values["B"] = st.number_input(
                "Beam Width B (mm)",
                value=200.0
            )


            input_values["D"] = st.number_input(
                "Beam Depth D (mm)",
                value=250.0
            )


            input_values["S"] = st.number_input(
                "Shear span S (mm)",
                value=1900.0
            )



        with col2:


            input_values["tp"] = st.number_input(
                "Steel Plate Thickness tp (mm)",
                value=2.3
            )


            input_values["S/L"] = st.number_input(
                "S/L Ratio",
                value=0.473684
            )


            input_values["b/B"] = st.number_input(
                "b/B Ratio",
                value=0.43
            )



        with col3:


            input_values["fc'"] = st.number_input(
                "Concrete Strength fc' (MPa)",
                value=40.0
            )


            input_values["Ec"] = st.number_input(
                "Concrete Elastic Modulus Ec (MPa)",
                value=22000.0
            )



        # Add fixed parameters

        for key,value in FIXED_PARAMETERS.items():

            input_values[key] = value





    # ========================================================
    # ADVANCED MODE
    # ========================================================


    else:


        st.subheader(
            "🔬 Advanced Mode"
        )


        st.info(
            "Enter all 20 input parameters."
        )



        col1,col2,col3,col4 = st.columns(4)



        with col1:


            input_values["B"] = st.number_input(
                "B",
                value=200.0
            )


            input_values["D"] = st.number_input(
                "D",
                value=250.0
            )


            input_values["S"] = st.number_input(
                "S",
                value=1900.0
            )


            input_values["fc'"] = st.number_input(
                "fc'",
                value=40.0
            )


            input_values["tp"] = st.number_input(
                "tp",
                value=2.3
            )




        with col2:


            input_values["S/L"] = st.number_input(
                "S/L",
                value=0.473684
            )


            input_values["b/B"] = st.number_input(
                "b/B",
                value=0.43
            )


            input_values["Enn"] = st.number_input(
                "Enn",
                value=144.0
            )


            input_values["Ess"] = st.number_input(
                "Ess",
                value=60.0
            )


            input_values["σn"] = st.number_input(
                "σn",
                value=4.0
            )




        with col3:


            input_values["Gc"] = st.number_input(
                "Gc",
                value=0.05
            )


            input_values["Ec"] = st.number_input(
                "Ec",
                value=22000.0
            )


            input_values["Ep"] = st.number_input(
                "Ep",
                value=125000.0
            )


            input_values["Er"] = st.number_input(
                "Er",
                value=165000.0
            )


            input_values["fyp"] = st.number_input(
                "fyp",
                value=300.0
            )




        with col4:


            input_values["fyr"] = st.number_input(
                "fyr",
                value=337.0
            )


            input_values["fup"] = st.number_input(
                "fup",
                value=375.0
            )


            input_values["fur"] = st.number_input(
                "fur",
                value=404.4
            )


            input_values["εup"] = st.number_input(
                "εup",
                value=0.2
            )


            input_values["εur"] = st.number_input(
                "εur",
                value=0.2
            )




    st.divider()



    # ========================================================
    # PREDICT BUTTON
    # ========================================================


    if st.button(
        "🚀 Predict Response",
        type="primary"
    ):



        with st.spinner(
            "Running AI prediction..."
        ):


            result = predict_single(
                input_values
            )



        st.success(
            "Prediction completed successfully!"
        )



        st.subheader(
            "Prediction Result"
        )



        result_df = pd.DataFrame(

            result.items(),

            columns=[
                "Parameter",
                "Value"
            ]

        )



        st.table(
            result_df
        )



        # Save current prediction for SHAP

        st.session_state["current_input"] = input_values

# ============================================================
# PART 3
# BATCH PREDICTION USING EXCEL FILE
# ============================================================


with tab_batch:


    st.header(
        "📂 Batch Prediction Using Excel"
    )


    st.write(
"""
Upload an Excel file containing multiple specimens.

For Simple Mode style Excel:
Required columns:

B, D, S, tp, S/L, b/B, fc', Ec

For Advanced Mode:
All 20 input parameters can be provided.
"""
    )


    st.divider()



    uploaded_file = st.file_uploader(

        "Upload Excel File (.xlsx)",

        type=["xlsx"]

    )



    if uploaded_file is not None:


        try:


            input_df = pd.read_excel(
                uploaded_file
            )



            st.success(

                f"{len(input_df)} samples loaded successfully"

            )



            st.subheader(
                "Input Data Preview"
            )


            st.dataframe(
                input_df.head()
            )



            if st.button(

                "🚀 Predict All Samples",

                type="primary"

            ):



                predictions = []


                progress_bar = st.progress(0)



                required_features = [

                    "B",
                    "D",
                    "S",
                    "fc'",
                    "tp",
                    "S/L",
                    "b/B",
                    "Enn",
                    "Ess",
                    "σn",
                    "Gc",
                    "Ec",
                    "Ep",
                    "Er",
                    "fyp",
                    "fyr",
                    "fup",
                    "fur",
                    "εup",
                    "εur"

                ]



                total_samples = len(input_df)



                for index,row in input_df.iterrows():



                    sample = {}



                    for feature in required_features:



                        if feature in row.index:


                            sample[feature] = row[feature]



                        else:


                            # Assign fixed values
                            # for simple mode Excel


                            sample[feature] = FIXED_PARAMETERS.get(

                                feature,

                                0

                            )



                    try:


                        prediction = predict_single(
                            sample
                        )


                    except Exception as e:


                        prediction = {

                            "Error":
                            str(e)

                        }



                    predictions.append(
                        prediction
                    )



                    progress_bar.progress(

                        int(
                            (index+1)
                            /
                            total_samples
                            *
                            100
                        )

                    )




                prediction_df = pd.DataFrame(
                    predictions
                )



                final_df = pd.concat(

                    [

                        input_df.reset_index(drop=True),

                        prediction_df

                    ],

                    axis=1

                )



                st.success(
                    "All predictions completed!"
                )



                st.subheader(
                    "Prediction Results"
                )



                st.dataframe(
                    final_df
                )



                # =================================================
                # SAVE EXCEL
                # =================================================


                output_file = (

                    "Debonding_AI_Prediction_Result.xlsx"

                )



                final_df.to_excel(

                    output_file,

                    index=False

                )



                with open(

                    output_file,

                    "rb"

                ) as file:



                    st.download_button(

                        label=
                        "⬇️ Download Prediction Excel",


                        data=file,


                        file_name=
                        output_file,


                        mime=
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                    )



        except Exception as e:


            st.error(

                f"Excel processing error: {e}"

            )

# ============================================================
# PART 4
# SHAP-BASED EXPLAINABILITY
# ============================================================


with tab_shap:


    st.header(
        "📊 SHAP-Based Model Explainability"
    )


    st.write(
"""
This module explains how input parameters influence
the Extra Trees debonding classifier prediction.

Positive SHAP contribution:
→ increases probability of debonding

Negative SHAP contribution:
→ decreases probability of debonding
"""
    )



    st.divider()



    # ========================================================
    # GLOBAL FEATURE IMPORTANCE
    # ========================================================


    st.subheader(
        "🌍 Global Feature Importance"
    )



    try:


        # ----------------------------------------------------
        # Use training-like reference dataset
        # ----------------------------------------------------


        X_reference = pd.DataFrame(

            np.zeros(
                (
                    100,
                    len(classifier_features)
                )
            ),

            columns=classifier_features

        )



        X_reference_scaled = pd.DataFrame(

            classifier_scaler.transform(
                X_reference
            ),

            columns=classifier_features

        )



        shap_output = shap_explainer(

            X_reference_scaled

        )



        # SHAP latest format
        # shape:
        # samples x features x classes


        if len(shap_output.values.shape) == 3:


            shap_class = shap_output.values[:,:,1]


        else:


            shap_class = shap_output.values




        importance = np.abs(
            shap_class
        ).mean(axis=0)



        importance_df = pd.DataFrame(

            {

                "Feature":
                classifier_features,


                "Mean |SHAP|":
                importance

            }

        ).sort_values(

            "Mean |SHAP|",

            ascending=False

        )



        st.dataframe(
            importance_df
        )



        fig,ax = plt.subplots(

            figsize=(8,5)

        )


        ax.barh(

            importance_df["Feature"],

            importance_df["Mean |SHAP|"]

        )


        ax.invert_yaxis()


        ax.set_xlabel(
            "Mean |SHAP Value|"
        )


        ax.set_title(
            "Global Feature Importance"
        )


        st.pyplot(
            fig
        )



    except Exception as e:


        st.error(
            f"Global SHAP error: {e}"
        )



    st.divider()



    # ========================================================
    # CURRENT SAMPLE EXPLANATION
    # ========================================================


    st.subheader(
        "🔍 Explain Current Prediction"
    )



    if "current_input" not in st.session_state:


        st.info(
"""
Please perform a prediction first
from the Prediction tab.
"""
        )


    else:



        if st.button(
            "Generate SHAP Explanation"
        ):



            try:



                current_sample = pd.DataFrame(

                    [
                        st.session_state["current_input"]

                    ]

                )



                X_sample = current_sample[

                    classifier_features

                ]



                X_sample_scaled = pd.DataFrame(

                    classifier_scaler.transform(

                        X_sample

                    ),

                    columns=classifier_features

                )



                shap_result = shap_explainer(

                    X_sample_scaled

                )



                # Latest SHAP output

                if len(shap_result.values.shape)==3:


                    values = shap_result.values[0,:,1]

                    base_value = (

                        shap_result.base_values[0,1]

                    )


                else:


                    values = shap_result.values[0]

                    base_value = (

                        shap_result.base_values[0]

                    )




                explanation = shap.Explanation(

                    values=values,

                    base_values=base_value,

                    data=X_sample_scaled.iloc[0],

                    feature_names=classifier_features

                )



                fig = plt.figure(

                    figsize=(10,6)

                )



                shap.plots.waterfall(

                    explanation,

                    show=False

                )



                st.pyplot(
                    fig
                )



                st.success(
                    "SHAP explanation generated successfully!"
                )



            except Exception as e:


                st.error(

                    f"SHAP prediction explanation error: {e}"

                )
