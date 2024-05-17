from typing import Dict, Any, List
import joblib

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoImageProcessor, AutoFeatureExtractor

import albumentations as A
from albumentations.pytorch import ToTensorV2


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
        preprocess_type: str,
        pretrained_model_name: str,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        spectogram_size: int,
        augmentation_probability: float,
        augmentations: List[str],
        max_length: int,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.target_column_name = target_column_name
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.preprocess_type = preprocess_type
        if self.preprocess_type == "spectogram":
            self.data_encoder = AutoImageProcessor.from_pretrained(
                pretrained_model_name,
            )
        elif self.preprocess_type == "vectorize":
            self.data_encoder = AutoFeatureExtractor.from_pretrained(
                pretrained_model_name,
            )
        else:
            raise ValueError(f"Invalid preprocess_type: {self.preprocess_type}.")
        dataset = self.get_dataset()
        self.datas = dataset["datas"]
        self.labels = dataset["labels"]
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectogram_size = spectogram_size
        self.augmentation_probability = augmentation_probability
        self.augmentations = augmentations
        self.transform = self.get_transform()
        self.max_length = max_length

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
        if self.preprocess_type == "spectogram":
            stft = librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length)
            spectrogram = librosa.amplitude_to_db(np.abs(stft))
            data = self.transform(image=spectrogram)["image"]
            encoded = self.encode_spectogram(data)
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

    def get_transform(self) -> A.Compose:
        transforms = [
            A.Resize(
                width=self.spectogram_size, height=self.spectogram_size, interpolation=2
            ),
        ]
        if self.split in ["train", "val"]:
            for aug in self.augmentations:
                if aug == "rotate30":
                    transforms.append(
                        A.Rotate(
                            limit=[30, 30],
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "rotate45":
                    transforms.append(
                        A.Rotate(
                            limit=[45, 45],
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "rotate90":
                    transforms.append(
                        A.Rotate(
                            limit=[90, 90],
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "hflip":
                    transforms.append(
                        A.HorizontalFlip(
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "vflip":
                    transforms.append(
                        A.VerticalFlip(
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "noise":
                    transforms.append(
                        A.GaussNoise(
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "blur":
                    transforms.append(
                        A.Blur(
                            blur_limit=7,
                            p=self.augmentation_probability,
                        )
                    )
            transforms.append(ToTensorV2())
            return A.Compose(transforms)
        else:
            transforms.append(ToTensorV2())
            return A.Compose(transforms)

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
