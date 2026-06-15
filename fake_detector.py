"""
fake_detector.py — perplexity-based AI text detector.
Uses GPT-2 to compute perplexity: low perplexity = suspiciously smooth = likely AI.
"""

import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_NAME = "distilgpt2"  # ~320MB, fast enough for Streamlit Cloud

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
        _model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def compute_perplexity(text: str) -> float:
    """Return perplexity of the text under distilgpt2."""
    model, tokenizer = _load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # cross-entropy loss

    return math.exp(loss.item())


def detect_ai_text(text: str) -> dict:
    """
    Analyze text and return detection result.

    Returns dict with:
      - score: float 0-100 (probability of AI generation)
      - verdict: str ("likely AI", "uncertain", "likely human")
      - perplexity: float
      - tags: list of str (short explanation labels)
    """
    if not text or len(text.strip()) < 20:
        return {
            "score": None,
            "verdict": "Text too short",
            "perplexity": None,
            "tags": ["need at least 20 characters"],
        }

    perplexity = compute_perplexity(text)

    # Calibration for political texts:
    # AI-generated text typically has perplexity 20-60
    # Human-written text typically has perplexity 80-300+
    if perplexity < 40:
        score = 90
        verdict = "likely AI-generated"
        tags = ["very low perplexity", "unusually smooth", "likely AI-generated"]
    elif perplexity < 70:
        score = 70
        verdict = "likely AI-generated"
        tags = ["low perplexity", "smooth text", "probably AI-generated"]
    elif perplexity < 100:
        score = 45
        verdict = "uncertain"
        tags = ["moderate perplexity", "borderline", "unclear signal"]
    elif perplexity < 150:
        score = 25
        verdict = "likely human-written"
        tags = ["higher perplexity", "natural variation", "probably human"]
    else:
        score = 10
        verdict = "likely human-written"
        tags = ["high perplexity", "irregular style", "likely human"]

    return {
        "score": score,
        "verdict": verdict,
        "perplexity": round(perplexity, 1),
        "tags": tags,
    }
