import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


path = "/data/kaggle-bird-clef-2024/data"

df = pd.read_csv(f"{path}/train.csv")
labels = df["label"].unique()

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

joblib.dump(label_encoder, f"{path}/label_encoder.pkl")

label_mapping = dict(zip(encoded_labels, labels))
joblib.dump(label_mapping, f"{path}/label_mapping.pkl")

print(label_encoder)
print(label_mapping)
