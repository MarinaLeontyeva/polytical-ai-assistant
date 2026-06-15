"""
fake_detector.py — AI text detector.
Uses roberta-base fine-tuned specifically to detect AI-generated text.
Model: Hello-SimpleAI/chatgpt-detector-roberta
Works well for English. Inspired by MELD paper (Li et al., 2026).
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
