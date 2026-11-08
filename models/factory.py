from models.classifier import BDHClassifier
from models.baselines import SimpleCNN, SimpleViT


def build_model(model_type, **cfg):
    if model_type == "bdh":
        return BDHClassifier(**cfg)
    if model_type == "cnn":
        return SimpleCNN(**cfg)
    if model_type == "vit":
        return SimpleViT(**cfg)
    raise ValueError(f"unknown model_type {model_type}")
