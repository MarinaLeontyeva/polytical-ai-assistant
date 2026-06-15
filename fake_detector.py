"""
fake_detector.py — AI text detector.
Uses roberta-base fine-tuned to detect AI-generated text.
Model: Hello-SimpleAI/chatgpt-detector-roberta
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def detect_ai_text(text: str) -> dict:
    if not text or len(text.strip()) < 20:
        return {
            "score": None,
            "verdict": "Text too short",
            "perplexity": None,
            "tags": ["need at least 20 characters"],
        }

    model, tokenizer = _load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    ai_score = round(probs[0][1].item() * 100)

    if ai_score >= 70:
        verdict = "likely AI-generated"
        tags = ["classifier confidence high", "AI patterns detected", "likely generated"]
    elif ai_score >= 40:
        verdict = "uncertain"
        tags = ["mixed signals", "borderline", "unclear signal"]
    else:
        verdict = "likely human-written"
        tags = ["human patterns", "natural variation", "probably authentic"]

    return {
        "score": ai_score,
        "verdict": verdict,
        "perplexity": None,
        "tags": tags,
    }
