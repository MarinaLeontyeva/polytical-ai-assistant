"""
fake_detector.py — dual AI text detector.
Method 1: perplexity-based (distilgpt2)
Method 2: RoBERTa classifier (chatgpt-detector-roberta)
"""

import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import pipeline as hf_pipeline

# ── Model 1: perplexity ──────────────────────────────────────
_gpt2_model = None
_gpt2_tokenizer = None

def _load_gpt2():
    global _gpt2_model, _gpt2_tokenizer
    if _gpt2_model is None:
        _gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
        _gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        _gpt2_model.eval()
    return _gpt2_model, _gpt2_tokenizer

def compute_perplexity(text: str) -> float:
    model, tokenizer = _load_gpt2()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())

def _perplexity_score(perplexity: float) -> int:
    if perplexity < 5:
        return 90
    elif perplexity < 10:
        return 70
    elif perplexity < 20:
        return 45
    elif perplexity < 40:
        return 25
    else:
        return 10

# ── Model 2: RoBERTa classifier ──────────────────────────────
_roberta = None

def _load_roberta():
    global _roberta
    if _roberta is None:
        _roberta = hf_pipeline(
            "text-classification",
            model="Hello-SimpleAI/chatgpt-detector-roberta",
            truncation=True,
            max_length=512,
        )
    return _roberta

def _roberta_score(text: str) -> int:
    classifier = _load_roberta()
    result = classifier(text)[0]
    label = result["label"]
    confidence = result["score"]
    if label == "ChatGPT":
        return round(confidence * 100)
    else:
        return round((1 - confidence) * 100)

# ── Combined detector ────────────────────────────────────────
def detect_ai_text(text: str) -> dict:
    if not text or len(text.strip()) < 20:
        return {
            "perplexity_score": None,
            "roberta_score": None,
            "combined_score": None,
            "verdict": "Text too short",
            "perplexity_value": None,
        }

    perplexity = compute_perplexity(text)
    p_score = _perplexity_score(perplexity)
    r_score = _roberta_score(text)

    # Итоговый счёт: roberta весит больше (она точнее)
    combined = round(p_score * 0.35 + r_score * 0.65)

    if combined >= 65:
        verdict = "likely AI-generated"
    elif combined >= 40:
        verdict = "uncertain"
    else:
        verdict = "likely human-written"

    return {
        "perplexity_score": p_score,
        "roberta_score": r_score,
        "combined_score": combined,
        "verdict": verdict,
        "perplexity_value": round(perplexity, 1),
    }
