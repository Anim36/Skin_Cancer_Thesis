"""Inference: single-image and batch prediction."""

from inference.predict import SkinCancerPredictor, PredictionResult, DISCLAIMER
from inference.predict_batch import batch_predict

__all__ = ["SkinCancerPredictor", "PredictionResult", "DISCLAIMER", "batch_predict"]
