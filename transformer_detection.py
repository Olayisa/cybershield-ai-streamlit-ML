from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MODEL_ID = "cybersectony/phishing-email-detection-distilbert_v2.4.1"

_LABEL_NAMES = {
    "LABEL_0": "Legitimate email",
    "LABEL_1": "Phishing email",
    "LABEL_2": "Legitimate URL",
    "LABEL_3": "Phishing URL",
}
_RISKY_LABELS = {"LABEL_1", "LABEL_3"}


class TransformerDetectionError(RuntimeError):
    """Raised when hosted transformer inference is unavailable or malformed."""


@dataclass(frozen=True)
class TransformerResult:
    label: str
    confidence: float
    risk_score: float
    scores: dict[str, float]
    model_id: str = MODEL_ID


def _prepare_text(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        raise ValueError("Email text cannot be empty.")

    # Preserve the opening context and message ending within DistilBERT's window.
    if len(cleaned) > 2000:
        cleaned = f"{cleaned[:1500]}\n...[content shortened]...\n{cleaned[-500:]}"
    return cleaned


def _parse_scores(raw_output: Any) -> dict[str, float]:
    items = raw_output
    if isinstance(items, list) and len(items) == 1 and isinstance(items[0], list):
        items = items[0]
    if not isinstance(items, list):
        items = [items]

    scores: dict[str, float] = {}
    for item in items:
        if isinstance(item, dict):
            label = item.get("label")
            score = item.get("score")
        else:
            label = getattr(item, "label", None)
            score = getattr(item, "score", None)
        if label is not None and score is not None:
            scores[str(label).upper()] = float(score)

    if not scores:
        raise TransformerDetectionError(
            "The transformer provider returned no classification scores."
        )
    return scores


def analyze_email_with_transformer(
    text: str,
    token: str,
    model_id: str = MODEL_ID,
) -> TransformerResult:
    """Classify email text through a hosted DistilBERT inference provider."""
    if not token or not str(token).strip():
        raise TransformerDetectionError(
            "HF_TOKEN is not configured in Streamlit secrets."
        )

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(
            provider="hf-inference",
            api_key=str(token).strip(),
        )
        raw_output = client.text_classification(
            _prepare_text(text),
            model=model_id,
            top_k=4,
        )
    except Exception as exc:
        raise TransformerDetectionError(
            "Transformer inference is temporarily unavailable. "
            "The existing ML and anomaly engines are still active."
        ) from exc

    scores = _parse_scores(raw_output)
    top_label, confidence = max(scores.items(), key=lambda pair: pair[1])
    risk_score = min(
        max(sum(score for label, score in scores.items() if label in _RISKY_LABELS), 0.0),
        1.0,
    )
    display_scores = {
        _LABEL_NAMES.get(label, label): score
        for label, score in scores.items()
    }

    return TransformerResult(
        label=_LABEL_NAMES.get(top_label, top_label),
        confidence=float(confidence),
        risk_score=float(risk_score),
        scores=display_scores,
        model_id=model_id,
    )
