#!/bin/bash

is_tuned="untuned"
strategy="ddp"
preprocess_type="vectorize"
upload_user="superb"
model_type="wav2vec2-base-superb-sid"
precision=32
batch_size=24
epochs="9 10"

for epoch in $epochs
do
    python main.py mode=predict \
        is_tuned=$is_tuned \
        strategy=$strategy \
        preprocess_type=$preprocess_type \
        upload_user=$upload_user \
        model_type=$model_type \
        precision=$precision \
        batch_size=$batch_size
done
