"""Lightweight anomaly-risk engines for CyberShield AI.

Developed and maintained by Yisa R. O. Adams.

The email engine compares behavioral features with a deterministic reference
baseline using Isolation Forest. Website batch scores are relative to the
uploaded batch. These scores measure unusualness, not attack probability.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


SUSPICIOUS_TERMS = (
    "account suspended", "act now", "confirm identity", "click immediately",
    "gift card", "password", "reset account", "security alert", "verify account",
    "wire transfer", "crypto payment", "unusual activity",
)
URGENCY_TERMS = (
    "urgent", "immediately", "final warning", "within 24 hours", "expires today",
    "action required", "do not ignore", "last chance",
)
CREDENTIAL_TERMS = (
    "password", "passcode", "login", "sign in", "credential", "verification code",
    "social security", "bank account",
)
DANGEROUS_EXTENSIONS = (
    ".exe", ".scr", ".js", ".vbs", ".bat", ".cmd", ".ps1", ".iso", ".lnk",
    ".docm", ".xlsm",
)
FEATURE_NAMES = (
    "log_message_length", "log_word_count", "link_count", "suspicious_term_count",
    "uppercase_ratio", "exclamation_count", "credential_term_count",
    "urgency_term_count", "ip_url_count", "dangerous_attachment_count",
    "currency_symbol_count", "digit_ratio",
)


@dataclass(frozen=True)
class AnomalyResult:
    risk_score: float
    percentile: float
    risk_level: str
    is_outlier: bool
    indicators: tuple[str, ...]
    features: dict[str, float]


def _count_terms(text: str, terms: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(term) for term in terms)


def extract_email_features(text: str) -> dict[str, float]:
    """Extract privacy-safe behavioral features without retaining message text."""
    text = text or ""
    words = re.findall(r"\b[\w'-]+\b", text)
    letters = [character for character in text if character.isalpha()]
    uppercase = sum(character.isupper() for character in letters)
    digits = sum(character.isdigit() for character in text)
    links = re.findall(r"(?:https?://|www\.)[^\s<>()]+", text, flags=re.I)
    ip_urls = 0
    for link in links:
        candidate = link if "://" in link else f"https://{link}"
        hostname = urlparse(candidate).hostname or ""
        if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", hostname):
            ip_urls += 1

    return {
        "log_message_length": math.log1p(len(text)),
        "log_word_count": math.log1p(len(words)),
        "link_count": float(len(links)),
        "suspicious_term_count": float(_count_terms(text, SUSPICIOUS_TERMS)),
        "uppercase_ratio": uppercase / max(len(letters), 1),
        "exclamation_count": float(text.count("!")),
        "credential_term_count": float(_count_terms(text, CREDENTIAL_TERMS)),
        "urgency_term_count": float(_count_terms(text, URGENCY_TERMS)),
        "ip_url_count": float(ip_urls),
        "dangerous_attachment_count": float(
            sum(text.lower().count(extension) for extension in DANGEROUS_EXTENSIONS)
        ),
        "currency_symbol_count": float(sum(text.count(symbol) for symbol in "$£€₦¥")),
        "digit_ratio": digits / max(len(text), 1),
    }


def _reference_email_baseline(seed: int = 42, rows: int = 1200) -> np.ndarray:
    """Create a stable, legitimate-like behavioral reference distribution."""
    rng = np.random.default_rng(seed)
    # Include short operational notes as well as longer business messages.
    word_count = np.maximum(rng.lognormal(mean=3.35, sigma=0.90, size=rows), 5)
    message_length = word_count * rng.normal(5.8, 0.65, rows).clip(3.5, 8.5)
    link_count = rng.choice([0, 1, 2], rows, p=[0.73, 0.24, 0.03])
    suspicious = rng.choice([0, 1], rows, p=[0.985, 0.015])
    uppercase_ratio = rng.beta(1.5, 38, rows)
    exclamations = rng.choice([0, 1, 2], rows, p=[0.82, 0.16, 0.02])
    credential = rng.choice([0, 1], rows, p=[0.992, 0.008])
    urgency = rng.choice([0, 1], rows, p=[0.975, 0.025])
    ip_urls = np.zeros(rows)
    dangerous = np.zeros(rows)
    currency = rng.choice([0, 1, 2], rows, p=[0.96, 0.035, 0.005])
    digit_ratio = rng.beta(1.2, 28, rows)

    return np.column_stack(
        (
            np.log1p(message_length), np.log1p(word_count), link_count, suspicious,
            uppercase_ratio, exclamations, credential, urgency, ip_urls, dangerous,
            currency, digit_ratio,
        )
    )


_EMAIL_BASELINE = _reference_email_baseline()
_EMAIL_ANOMALY_MODEL = IsolationForest(
    n_estimators=300,
    contamination=0.04,
    random_state=42,
    n_jobs=-1,
).fit(_EMAIL_BASELINE)
_BASELINE_SCORES = _EMAIL_ANOMALY_MODEL.score_samples(_EMAIL_BASELINE)


def _risk_level(percentile: float) -> str:
    if percentile >= 97:
        return "Critical"
    if percentile >= 90:
        return "High"
    if percentile >= 75:
        return "Elevated"
    return "Low"


def analyze_email_anomaly(text: str) -> AnomalyResult:
    features = extract_email_features(text)
    row = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=float)
    raw_score = float(_EMAIL_ANOMALY_MODEL.score_samples(row)[0])
    percentile = float(100 * np.mean(_BASELINE_SCORES >= raw_score))
    risk_score = percentile / 100

    indicators: list[str] = []
    if features["suspicious_term_count"]:
        indicators.append("Suspicious or social-engineering language")
    if features["credential_term_count"]:
        indicators.append("Credential or identity request")
    if features["urgency_term_count"]:
        indicators.append("Urgency or pressure language")
    if features["link_count"] >= 2:
        indicators.append("Multiple links")
    if features["ip_url_count"]:
        indicators.append("IP-address link")
    if features["dangerous_attachment_count"]:
        indicators.append("Potentially dangerous attachment type")
    if features["uppercase_ratio"] >= 0.25:
        indicators.append("Unusually high uppercase ratio")
    if features["exclamation_count"] >= 3:
        indicators.append("Unusually heavy exclamation use")
    if not indicators:
        indicators.append("No dominant rule-based indicator; score reflects overall pattern")

    return AnomalyResult(
        risk_score=risk_score,
        percentile=percentile,
        risk_level=_risk_level(percentile),
        is_outlier=bool(_EMAIL_ANOMALY_MODEL.predict(row)[0] == -1),
        indicators=tuple(indicators),
        features=features,
    )


def score_tabular_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """Return relative Isolation Forest anomaly scores for a website-feature batch."""
    if len(frame) < 10:
        raise ValueError("At least 10 records are required for relative batch anomaly scoring.")
    if len(frame) > 5000:
        raise ValueError("Use 5,000 records or fewer per anomaly-scoring batch.")

    prepared = frame.copy()
    for column in prepared.columns:
        numeric = pd.to_numeric(prepared[column], errors="coerce")
        if numeric.notna().mean() >= 0.8:
            prepared[column] = numeric.fillna(numeric.median())
        else:
            prepared[column] = prepared[column].fillna("missing").astype(str)

    matrix = pd.get_dummies(prepared, dummy_na=True, dtype=float)
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    raw_scores = model.fit(matrix).score_samples(matrix)
    ranks = pd.Series(-raw_scores).rank(method="average", pct=True).to_numpy() * 100

    return pd.DataFrame(
        {
            "Anomaly_Percentile": ranks,
            "Anomaly_Risk": [_risk_level(value) for value in ranks],
            "Anomaly_Outlier": model.predict(matrix) == -1,
        },
        index=frame.index,
    )
