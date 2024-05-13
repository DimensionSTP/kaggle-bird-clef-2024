from typing import List, Dict
import os
import glob as glob

import pandas as pd


def read_file(
    audio_path: str,
    categories_list: List[str],
) -> Dict[str, List[str]]:
    audios = []
    categories = []
    for category in categories_list:
        audio_paths = glob.glob(
            os.path.join(audio_path, category, "*.ogg"),
            recursive=True,
        )
        print(f"found {len(audio_paths)} file in train_audios/{category} folder.")
        for audio in audio_paths:
            audios.append(audio.split("/")[-1])
            categories.append(category)
    print(f"total {len(audios)} file in audios folder.")
    return {
        "audios": audios,
        "categories": categories,
    }


if __name__ == "__main__":
    DATA_PATH = "/data/kaggle-bird-clef-2024/data"
    AUDIO_PATH = f"{DATA_PATH}/train_audio"
    categories_list = os.listdir(AUDIO_PATH)

    output = read_file(
        AUDIO_PATH,
        categories_list,
    )
    audios = output["audios"]
    categories = output["categories"]

    data = pd.DataFrame(
        {
            "audio_path": audios,
            "label": categories,
        }
    )
    csv_path = f"{DATA_PATH}/train.csv"
    data.to_csv(csv_path, index=False)
