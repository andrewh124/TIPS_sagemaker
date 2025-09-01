import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


class IdeaDetectionPipeline:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification(model_dir)
        with open(Path(model_dir, "labels.json"), "r") as f:
            self.labels = json.load(f)

    def get_predictions(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt")
        logits = self.model(**inputs).logits

        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        preds = self.map_probs_to_labels(probs)

        return {
            "logits": logits.tolist(),
            "label_probabilities": probs.tolist(),
            "words": self.tokenizer.tokenize(text),
            "predicted_labels": preds,
        }

    def map_probs_to_labels(self, probs: np.ndarray) -> list[list[str]]:
        indices = np.argwhere(probs > 0.5)
        predicted_labels = [[] for word in range(probs.shape[1])]

        for word_index, label_index in indices:
            label = self.labels[str(label_index)]
            predicted_labels[word_index].append(label)

        return predicted_labels
