from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "trained_models"
EMAIL_MODEL_PATH = MODEL_DIR / "best_email_security_model.joblib"
PHISHING_MODEL_PATH = MODEL_DIR / "best_phishing_website_model.joblib"
RESULTS_PATH = MODEL_DIR / "model_performance_results.csv"


st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {background: #07111f; color: #eef6ff;}
    [data-testid="stSidebar"] {background: #0a192b;}
    .hero {
        padding: 2rem; border-radius: 22px;
        background: linear-gradient(135deg, #0b2742 0%, #102f39 55%, #12372c 100%);
        border: 1px solid rgba(100, 232, 194, .25);
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2.6rem; color: #f5fbff;}
    .hero p {color: #bed4e8; font-size: 1.05rem; margin: .65rem 0 0;}
    .metric-card {
        background: #0d2035; border: 1px solid #1b405d;
        border-radius: 16px; padding: 1.1rem; min-height: 120px;
    }
    .metric-card h3 {color: #65e6c3; margin-top: 0;}
    .safe-box {
        background: rgba(34, 197, 94, .12); border: 1px solid rgba(34, 197, 94, .45);
        padding: 1rem; border-radius: 14px;
    }
    .danger-box {
        background: rgba(239, 68, 68, .13); border: 1px solid rgba(239, 68, 68, .5);
        padding: 1rem; border-radius: 14px;
    }
    .warning-box {
        background: rgba(245, 158, 11, .12); border: 1px solid rgba(245, 158, 11, .45);
        padding: 1rem; border-radius: 14px;
    }
    div[data-testid="stMetric"] {
        background: #0d2035; border: 1px solid #1b405d;
        padding: 1rem; border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_results(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def model_score(model: Any, X: Any, prediction: Any) -> tuple[float | None, dict[str, float]]:
    """Return predicted-class confidence and all available class scores."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = list(model.classes_)
        scores = {str(label): float(value) for label, value in zip(classes, probabilities)}
        return scores.get(str(prediction)), scores

    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X))
        if raw.ndim == 1:
            positive = 1 / (1 + np.exp(-raw[0]))
            classes = list(model.classes_)
            probabilities = [1 - positive, positive]
        else:
            values = raw[0] - np.max(raw[0])
            probabilities = np.exp(values) / np.exp(values).sum()
            classes = list(model.classes_)
        scores = {str(label): float(value) for label, value in zip(classes, probabilities)}
        return scores.get(str(prediction)), scores

    return None, {}


def is_risky_label(label: Any) -> bool:
    value = str(label).strip().lower()
    risky_terms = (
        "phish", "malicious", "spam", "fraud", "attack", "unsafe",
        "suspicious", "1", "-1", "bad", "threat",
    )
    safe_terms = ("legitimate", "benign", "safe", "normal", "ham", "0", "good")
    if any(term == value or term in value for term in safe_terms):
        return False
    return any(term == value or term in value for term in risky_terms)


def render_prediction(label: Any, confidence: float | None, scores: dict[str, float]) -> None:
    risky = is_risky_label(label)
    confidence_text = f"{confidence:.1%}" if confidence is not None else "Unavailable"
    css_class = "danger-box" if risky else "safe-box"
    heading = "Potential threat detected" if risky else "No threat detected"
    icon = "🚨" if risky else "✅"

    st.markdown(
        f'<div class="{css_class}"><h3>{icon} {heading}</h3>'
        f'<p><strong>Predicted class:</strong> {label}<br>'
        f'<strong>Model confidence:</strong> {confidence_text}</p></div>',
        unsafe_allow_html=True,
    )

    if scores:
        score_frame = pd.DataFrame(
            {"Class": list(scores.keys()), "Probability": list(scores.values())}
        ).sort_values("Probability", ascending=False)
        st.bar_chart(score_frame.set_index("Class"), horizontal=True)

    if risky:
        st.warning(
            "Do not click links, open attachments, disclose credentials, or send payment. "
            "Verify the sender through a trusted channel and report the message to your security team."
        )
    else:
        st.info(
            "The model did not detect a known threat pattern. Continue normal verification—"
            "machine-learning predictions are not a guarantee of safety."
        )


def model_status_card(name: str, model: Any, filename: str) -> None:
    if model is None:
        st.error(f"{name}: model missing")
        st.caption(f"Copy `{filename}` into `{MODEL_DIR}`.")
    else:
        st.success(f"{name}: ready")


email_model = load_model(EMAIL_MODEL_PATH)
phishing_model = load_model(PHISHING_MODEL_PATH)
performance_results = load_results(RESULTS_PATH)


with st.sidebar:
    st.markdown("## 🛡️ CyberShield AI")
    st.caption("Email and phishing threat detection")
    page = st.radio(
        "Navigation",
        ["Overview", "Email detector", "Website detector", "Model performance", "About"],
    )
    st.divider()
    st.markdown("#### System status")
    model_status_card("Email detector", email_model, EMAIL_MODEL_PATH.name)
    model_status_card("Website detector", phishing_model, PHISHING_MODEL_PATH.name)
    st.caption("Prepared by Yisa R. O. Adams")


if page == "Overview":
    st.markdown(
        """
        <div class="hero">
          <h1>CyberShield AI</h1>
          <p>An explainable machine-learning dashboard for suspicious-email classification
          and engineered phishing-feature detection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><h3>Email analysis</h3><p>TF-IDF language features classify suspicious and legitimate messages.</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><h3>Website analysis</h3><p>Batch-score structured UCI-style phishing indicators using your trained pipeline.</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><h3>Model evidence</h3><p>Review accuracy, precision, recall, and F1 results saved during training.</p></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Responsible use")
    st.markdown(
        """
        - Treat predictions as decision support, not a replacement for a security analyst.
        - Avoid submitting passwords, private keys, financial data, or sensitive client information.
        - Investigate false negatives carefully; in cybersecurity, missed threats can be costly.
        """
    )


elif page == "Email detector":
    st.title("📨 Suspicious email detector")
    st.write("Paste an email subject and body. The deployed NLP pipeline will classify the combined text.")

    if email_model is None:
        st.error(f"Model not found. Place `{EMAIL_MODEL_PATH.name}` inside `{MODEL_DIR}`.")
        st.stop()

    with st.form("email_form"):
        subject = st.text_input("Email subject", placeholder="Urgent: your account will be suspended")
        body = st.text_area(
            "Email body",
            height=230,
            placeholder="Paste the message here…",
        )
        submitted = st.form_submit_button("Analyze email", type="primary", use_container_width=True)

    if submitted:
        combined_text = f"Subject: {subject}\n\n{body}".strip()
        if len(combined_text.replace("Subject:", "").strip()) < 10:
            st.warning("Enter at least 10 characters of email content.")
        else:
            with st.spinner("Analyzing message patterns…"):
                prediction = email_model.predict([combined_text])[0]
                confidence, scores = model_score(email_model, [combined_text], prediction)
            render_prediction(prediction, confidence, scores)

    with st.expander("Try a safe demonstration message"):
        st.code("Subject: Project meeting\nHi team, our project meeting is Thursday at 10 AM. Please confirm availability.")
    with st.expander("Try a suspicious demonstration message"):
        st.code("Subject: Account suspended\nClick this link immediately and enter your password to restore access.")


elif page == "Website detector":
    st.title("🌐 Phishing website feature detector")
    st.write(
        "Upload a CSV containing the same feature columns used to train your website model. "
        "The target/label column is optional and will be ignored if present."
    )

    if phishing_model is None:
        st.error(f"Model not found. Place `{PHISHING_MODEL_PATH.name}` inside `{MODEL_DIR}`.")
        st.stop()

    expected_features = None
    if hasattr(phishing_model, "feature_names_in_"):
        expected_features = list(phishing_model.feature_names_in_)
        with st.expander(f"Expected input columns ({len(expected_features)})"):
            st.code("\n".join(expected_features))

    tab_upload, tab_json = st.tabs(["CSV batch scan", "Single JSON record"])

    with tab_upload:
        uploaded = st.file_uploader("Upload website-feature CSV", type=["csv"])
        if uploaded is not None:
            incoming = pd.read_csv(uploaded)
            st.write("Preview")
            st.dataframe(incoming.head(20), use_container_width=True)

            if st.button("Scan uploaded records", type="primary", use_container_width=True):
                model_input = incoming.copy()
                if expected_features is not None:
                    missing = [c for c in expected_features if c not in model_input.columns]
                    if missing:
                        st.error(f"Missing {len(missing)} required feature columns: {missing}")
                        st.stop()
                    model_input = model_input[expected_features]

                predictions = phishing_model.predict(model_input)
                output = incoming.copy()
                output["Predicted_Class"] = predictions

                if hasattr(phishing_model, "predict_proba"):
                    probabilities = phishing_model.predict_proba(model_input)
                    output["Prediction_Confidence"] = probabilities.max(axis=1)

                output["Risk_Flag"] = ["Threat" if is_risky_label(x) else "No threat" for x in predictions]
                threat_count = int((output["Risk_Flag"] == "Threat").sum())

                m1, m2, m3 = st.columns(3)
                m1.metric("Records scanned", len(output))
                m2.metric("Potential threats", threat_count)
                m3.metric("No threat detected", len(output) - threat_count)
                st.dataframe(output, use_container_width=True)
                st.download_button(
                    "Download scan results",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="cybershield_website_scan_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    with tab_json:
        st.caption("Use this option when you already have one engineered feature record.")
        default_json = {}
        if expected_features:
            default_json = {column: 0 for column in expected_features}
        json_text = st.text_area("Feature JSON", value=json.dumps(default_json, indent=2), height=300)
        if st.button("Analyze JSON record", use_container_width=True):
            try:
                record = json.loads(json_text)
                one_row = pd.DataFrame([record])
                if expected_features is not None:
                    missing = [c for c in expected_features if c not in one_row.columns]
                    if missing:
                        raise ValueError(f"Missing required features: {missing}")
                    one_row = one_row[expected_features]
                prediction = phishing_model.predict(one_row)[0]
                confidence, scores = model_score(phishing_model, one_row, prediction)
                render_prediction(prediction, confidence, scores)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                st.error(f"Could not analyze this record: {exc}")


elif page == "Model performance":
    st.title("📊 Model performance")
    if performance_results is None:
        st.warning(f"Place `{RESULTS_PATH.name}` inside `{MODEL_DIR}` to display metrics.")
    else:
        st.dataframe(performance_results, use_container_width=True)
        metric_columns = [c for c in ["Accuracy", "Precision", "Recall", "F1 Score"] if c in performance_results.columns]
        if metric_columns and "Model" in performance_results.columns:
            chart_data = performance_results.set_index("Model")[metric_columns]
            st.bar_chart(chart_data)
        if "F1 Score" in performance_results.columns:
            best = performance_results.loc[performance_results["F1 Score"].idxmax()]
            st.success(f"Best recorded F1 model: {best.get('Model', 'Unknown')} — {best['F1 Score']:.3f}")
    st.info("For threat detection, inspect recall and F1 alongside accuracy. Accuracy alone can hide missed attacks.")


else:
    st.title("About this project")
    st.markdown(
        """
        **CyberShield AI** is an end-to-end machine-learning portfolio application prepared by
        **Yisa R. O. Adams**. It operationalizes two trained pipelines:

        1. NLP email classification using TF-IDF and a linear classifier.
        2. Structured phishing-website classification using preprocessed UCI-style features.

        The application provides interactive predictions, confidence visualization, batch CSV
        scanning, downloadable results, model-performance reporting, and missing-model diagnostics.

        **Important:** This application is educational and provides decision support. It is not a
        substitute for endpoint protection, secure email gateways, browser isolation, threat
        intelligence, or professional incident response.
        """
    )

