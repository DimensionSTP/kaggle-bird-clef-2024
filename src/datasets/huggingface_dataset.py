from typing import Dict, Any, List
import joblib

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoImageProcessor, AutoFeatureExtractor


class KaggleBirdClefDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        target_column_name: str,
        num_devices: int,
        batch_size: int,
        sampling_rate: int,
        num_mels: int,
        max_length: int,
        pretrained_model_name: str,
        preprocess_type: str,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.target_column_name = target_column_name
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.max_length = max_length
        self.pretrained_model_name = pretrained_model_name
        self.preprocess_type = preprocess_type
        if "timm" in self.pretrained_model_name:
            self.data_encoder = None
        elif "ast" in self.pretrained_model_name:
            self.data_encoder = AutoFeatureExtractor.from_pretrained(
                self.pretrained_model_name,
                sampling_rate=self.sampling_rate,
                num_mel_bins=self.num_mels,
            )
        else:
            if self.preprocess_type == "spectogram":
                self.data_encoder = AutoImageProcessor.from_pretrained(
                    self.pretrained_model_name,
                )
            elif self.preprocess_type == "vectorize":
                self.data_encoder = AutoFeatureExtractor.from_pretrained(
                    self.pretrained_model_name,
                )
            else:
                raise ValueError(f"Invalid preprocess_type: {self.preprocess_type}.")
        dataset = self.get_dataset()
        self.datas = dataset["datas"]
        self.labels = dataset["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        data = librosa.load(
            self.datas[idx],
            sr=self.sampling_rate,
        )[0]
        mel_spectogram = librosa.feature.melspectrogram(
            y=data,
            sr=self.sampling_rate,
            n_mels=self.num_mels,
        )
        mel_spectogram_db = librosa.power_to_db(
            mel_spectogram,
            ref=np.max,
        )
        mel_spectogram_db = (mel_spectogram_db - mel_spectogram_db.min()) / (
            mel_spectogram_db.max() - mel_spectogram_db.min()
        )
        if mel_spectogram_db.shape[1] > self.max_length:
            mel_spectogram_db = mel_spectogram_db[:, : self.max_length]
        else:
            padding = self.max_length - mel_spectogram_db.shape[1]
            mel_spectogram_db = np.pad(
                mel_spectogram_db,
                (
                    (0, 0),
                    (0, padding),
                ),
                mode="constant",
            )
        spectogram = torch.tensor(mel_spectogram_db).unsqueeze(0)
        spectogram = spectogram.repeat(3, 1, 1)
        if "timm" in self.pretrained_model_name:
            encoded = {}
            encoded["encoded"] = spectogram
        elif "ast" in self.pretrained_model_name:
            encoded = self.encode_ast(data)
        else:
            if self.preprocess_type == "spectogram":
                encoded = self.encode_spectogram(spectogram)
            else:
                encoded = self.encode_audio(data)
        encoded["labels"] = torch.tensor(
            [self.labels[idx]],
            dtype=torch.long,
        ).squeeze(0)
        return {
            "encoded": encoded,
            "index": idx,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            csv_path = f"{self.data_path}/train.csv"
            data = pd.read_csv(csv_path)
            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
                stratify=data[self.target_column_name],
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        elif self.split == "test":
            csv_path = f"{self.data_path}/sample_submission.csv"
            data = pd.read_csv(csv_path)
        elif self.split == "predict":
            csv_path = f"{self.data_path}/sample_submission.csv"
            data = pd.read_csv(csv_path)
            if self.num_devices > 1:
                last_row = data.iloc[-1]
                total_batch_size = self.num_devices * self.batch_size
                remainder = (len(data) % total_batch_size) % self.num_devices
                if remainder != 0:
                    num_dummies = self.num_devices - remainder
                    repeated_rows = pd.DataFrame([last_row] * num_dummies)
                    repeated_rows.reset_index(
                        drop=True,
                        inplace=True,
                    )
                    data = pd.concat(
                        [
                            data,
                            repeated_rows,
                        ],
                        ignore_index=True,
                    )
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        if self.split == "train":
            datas = [file_path for file_path in data["audio_path"]]
        elif self.split == "val":
            datas = [file_path for file_path in data["audio_path"]]
        elif self.split == "test":
            datas = [file_path for file_path in data["audio_path"]]
        else:
            datas = [file_path for file_path in data["audio_path"]]
        str_labels = data[self.target_column_name].tolist()
        label_encoder = joblib.load(f"{self.data_path}/label_encoder.pkl")
        labels = label_encoder.transform(str_labels)
        return {
            "datas": datas,
            "labels": labels,
        }

    def encode_ast(
        self,
        data: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.data_encoder(
            data,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

    def encode_spectogram(
        self,
        data: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.data_encoder(
            data,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

    def encode_audio(
        self,
        data: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.data_encoder(
            data,
            sampling_rate=self.sampling_rate,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded
