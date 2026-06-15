"""
deepfake_detector.py — image deepfake detector.
Uses a pretrained ViT-based classifier from HuggingFace.
Model: Organika/sdxl-detector
"""

from PIL import Image
from transformers import pipeline as hf_pipeline

MODEL_NAME = "Organika/sdxl-detector"

_detector = None


def _load_detector():
    global _detector
    if _detector is None:
        _detector = hf_pipeline("image-classification", model=MODEL_NAME)
    return _detector


def detect_deepfake(image: Image.Image) -> dict:
    detector = _load_detector()
    results = detector(image)

    scores = {r["label"].lower(): r["score"] for r in results}
    ai_score = scores.get("artificial", scores.get("fake", 0.0))
    ai_percent = round(ai_score * 100)

    if ai_percent >= 70:
        verdict = "likely AI-generated / deepfake"
        tags = ["synthetic artifacts detected", "likely generated", "not authentic"]
    elif ai_percent >= 40:
        verdict = "uncertain"
        tags = ["mixed signals", "borderline case", "inconclusive"]
    else:
        verdict = "likely real photo"
        tags = ["natural image patterns", "low synthetic signal", "probably authentic"]

    return {
        "score": ai_percent,
        "verdict": verdict,
        "tags": tags,
        "raw": results,
    }
