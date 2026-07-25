# CyberShield AI

An end-to-end Streamlit cybersecurity application that combines supervised machine learning, NLP, and unsupervised anomaly-risk detection for suspicious emails and phishing website features.

## Current capabilities

- Phishing email classification using TF-IDF and a trained linear classifier
- Structured website-phishing classification using a trained scikit-learn pipeline
- Isolation Forest email anomaly-risk analysis using privacy-safe behavioral features
- Relative website-batch outlier detection for uploads containing 10–5,000 records
- Prediction confidence, anomaly percentile, behavioral indicators, and combined threat scoring
- Batch CSV scanning, downloadable results, and model-performance reporting

## Anomaly-risk interpretation

The anomaly percentile measures how unusual an input is relative to a reference baseline or uploaded batch. It is not an attack probability. The current release performs present-time anomaly-risk detection; genuine future-event forecasting requires timestamped scan history and time-based validation.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place the trained artifacts either in the repository root or in `trained_models/`:

```text
best_email_security_model.joblib
best_phishing_website_model.joblib
model_performance_results.csv
```

## Responsible use

CyberShield AI is an educational decision-support application. It does not replace secure email gateways, endpoint protection, threat intelligence, browser isolation, or professional incident response. Do not submit passwords, private keys, or confidential client data.

Developed and maintained by **Yisa R. O. Adams**.
