"""
fake_detector.py — perplexity-based AI text detector using distilgpt2.
Low perplexity = text too smooth and predictable = likely AI-generated.
Inspired by MELD paper (Li et al., 2026).
"""

import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_NAME = "distilgpt2"

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
    model, tokenizer = _load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())


def detect_ai_text(text: str) -> dict:
    if not text or len(text.strip()) < 20:
        return {
            "score": None,
            "verdict": "Text too short",
            "perplexity": None,
            "tags": ["need at least 20 characters"],
        }

    perplexity = compute_perplexity(text)

    if perplexity < 30:
        score = 85
        verdict = "likely AI-generated"
        tags = ["very low perplexity", "unusually smooth", "likely AI-generated"]
    elif perplexity < 55:
        score = 65
        verdict = "likely AI-generated"
        tags = ["low perplexity", "smooth text", "probably AI-generated"]
    elif perplexity < 90:
        score = 40
        verdict = "uncertain"
        tags = ["moderate perplexity", "borderline", "unclear signal"]
    elif perplexity < 160:
        score = 20
        verdict = "likely human-written"
        tags = ["higher perplexity", "natural variation", "probably human"]
    else:
        score = 5
        verdict = "likely human-written"
        tags = ["high perplexity", "irregular style", "likely human"]

    return {
        "score": score,
        "verdict": verdict,
        "perplexity": round(perplexity, 1),
        "tags": tags,
    }
