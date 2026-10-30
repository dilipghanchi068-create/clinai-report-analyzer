
import streamlit as st
import joblib
import re
from pathlib import Path

st.set_page_config(
    page_title="ClinAI - Clinical Report Analyzer",
    page_icon="📋",
    layout="centered"
)

st.title("📋 ClinAI – Clinical Report Analyzer")
st.write(
    "Upload a clinical report in TXT format "
    "or paste the report text below."
)

# Load the trained machine-learning model
@st.cache_resource
def load_model():
    model_path = Path("clinical_report_classifier.joblib")
    if not model_path.exists():
        raise FileNotFoundError(
            "clinical_report_classifier.joblib not found."
        )
    return joblib.load(model_path)

# Supported medical terms for text matching
MEDICAL_TERMS = [
    "hemoglobin", "glucose", "blood glucose",
    "creatinine", "cholesterol", "triglycerides",
    "platelet count", "white blood cell count",
    "wbc", "rbc", "sodium", "potassium",
    "complete blood count", "cbc", "fever",
    "cough", "fatigue", "blood pressure",
    "diabetes", "infection", "anemia"
]

# Extract recognized terms
def extract_terms(text):
    found = []
    lower_text = text.lower()

    for term in MEDICAL_TERMS:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, lower_text):
            found.append(term.title())

    return found

# Extract simple test-value-unit patterns
def extract_measurements(text):
    pattern = re.compile(
        r"\b(hemoglobin|glucose|creatinine|cholesterol|"
        r"triglycerides|platelet count|white blood cell count|"
        r"sodium|potassium)\b"
        r"\s*[:=]?\s*"
        r"([+-]?\d+(?:\.\d+)?)\s*"
        r"(mg/dL|g/dL|mmol/L|mmol|mEq/L|%|"
        r"mg/L|g/L|U/L|mmHg|bpm)?",
        re.IGNORECASE
    )

    results = []
    for match in pattern.finditer(text):
        results.append({
            "Test": match.group(1).title(),
            "Value": match.group(2),
            "Unit": match.group(3) or "Not specified"
        })

    return results

# Upload a TXT file
uploaded_file = st.file_uploader(
    "Upload clinical report (.txt)",
    type=["txt"]
)

report_text = ""

if uploaded_file is not None:
    try:
        report_text = uploaded_file.getvalue().decode(
            "utf-8-sig"
        )
        st.success("TXT report uploaded successfully.")
        with st.expander("Preview uploaded report"):
            st.text(report_text[:5000])
    except UnicodeDecodeError:
        st.error("Could not read this file as UTF-8 text.")
        report_text = ""

# Allow manual text entry too
manual_text = st.text_area(
    "Or paste your clinical report here",
    height=150,
    placeholder="Enter clinical report text..."
)

if manual_text.strip():
    report_text = manual_text.strip()

if st.button("Analyze Report", type="primary"):
    if not report_text.strip():
        st.warning("Please upload a TXT file or enter report text.")
    else:
        try:
            with st.spinner("Analyzing report..."):
                model = load_model()
                category = model.predict([report_text])[0]
                terms = extract_terms(report_text)
                measurements = extract_measurements(report_text)

            st.subheader("Analysis Result")
            st.write("**Predicted Report Category:**", str(category))

            st.write("**Medical Terms Found:**")
            if terms:
                st.write(", ".join(terms))
            else:
                st.info("No matching terms found in the supported list.")

            st.write("**Laboratory Measurements:**")
            if measurements:
                st.dataframe(
                    measurements,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No supported test-value pairs were detected.")

            sentences = re.split(
                r"(?<=[.!?])\s+|\n+",
                report_text.strip()
            )
            sentences = [s.strip() for s in sentences if s.strip()]

            st.write("**Report Summary:**")
            st.write(" ".join(sentences[:3]))

            st.info(
                "Educational prototype only. Extracted information "
                "may be incomplete or incorrect. This app does not "
                "provide a medical diagnosis or treatment advice."
            )

        except Exception as error:
            st.error(
                "Analysis failed. Check the model file and its "
                "compatibility with the input text."
            )
            st.caption(f"Technical detail: {error}")
