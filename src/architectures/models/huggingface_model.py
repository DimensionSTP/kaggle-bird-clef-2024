from typing import Dict

import torch
from torch import nn

from transformers import (
    AutoModelForImageClassification,
    Wav2Vec2ForSequenceClassification,
)


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        preprocess_type: str,
        pretrained_model_name: str,
        num_labels: int,
    ) -> None:
        super().__init__()
        if preprocess_type == "spectogram":
            self.model = AutoModelForImageClassification.from_pretrained(
                pretrained_model_name,
                num_labels=num_labels,
                output_hidden_states=False,
                ignore_mismatched_sizes=True,
            )
        elif preprocess_type == "vectorize":
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                pretrained_model_name,
                num_labels=num_labels,
                output_hidden_states=False,
                ignore_mismatched_sizes=True,
            )
        else:
            raise ValueError(f"Invalid preprocess_type: {preprocess_type}.")

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output = self.model(**encoded)
        return output
