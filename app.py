
import streamlit as st
import joblib
import re

st.set_page_config(page_title="ClinAI", page_icon="🏥")

st.title("🏥 ClinAI – Clinical Report Analyzer")
st.write("Analyze a clinical report using Machine Learning.")

@st.cache_resource
def load_model():
    return joblib.load("clinical_report_classifier.joblib")

try:
    model = load_model()
except Exception:
    st.error("Model file not found. Please upload the trained model.")
    st.stop()

report = st.text_area(
    "Paste your clinical report here",
    height=200,
    placeholder="Enter clinical report text..."
)

if st.button("Analyze Report"):
    if report.strip():
        prediction = model.predict([report])[0]
        st.subheader("Analysis Result")
        st.write("**Predicted Report Category:**", str(prediction))

        terms = re.findall(
            r"\b(?:hemoglobin|glucose|platelets|cholesterol|creatinine|blood pressure|x-ray|MRI|CT scan)\b",
            report,
            re.IGNORECASE
        )

        st.write("**Medical terms found:**")
        st.write(", ".join(sorted(set(terms), key=str.lower)) if terms else "No matching terms found.")

        st.info("Educational prototype only. This result is not a medical diagnosis.")
    else:
        st.warning("Please enter a clinical report first.")
