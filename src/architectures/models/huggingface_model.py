from typing import Dict

import torch
from torch import nn

import timm

from transformers import (
    AutoModelForImageClassification,
    AutoModelForAudioClassification,
)


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        preprocess_type: str,
        num_labels: int,
    ) -> None:
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        if "timm" in self.pretrained_model_name:
            self.model = timm.create_model(
                self.pretrained_model_name[5:],
                pretrained=True,
                num_classes=num_labels,
            )
        elif "ast" in self.pretrained_model_name:
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.pretrained_model_name,
                num_labels=num_labels,
                output_hidden_states=False,
                ignore_mismatched_sizes=True,
            )
        else:
            if preprocess_type == "spectogram":
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.pretrained_model_name,
                    num_labels=num_labels,
                    output_hidden_states=False,
                    ignore_mismatched_sizes=True,
                )
            elif preprocess_type == "vectorize":
                self.model = AutoModelForAudioClassification.from_pretrained(
                    self.pretrained_model_name,
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
        if "timm" in self.pretrained_model_name:
            output = self.model(encoded["encoded"])
        else:
            output = self.model(**encoded)
        return output
