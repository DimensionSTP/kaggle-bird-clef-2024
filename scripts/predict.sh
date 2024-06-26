#!/bin/bash

is_tuned="untuned"
strategy="ddp"
preprocess_type="vectorize"
upload_user="superb"
model_type="wav2vec2-base-superb-sid"
precision=32
batch_size=512
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

is_tuned="untuned"
strategy="ddp"
preprocess_type="spectogram"
upload_user="microsoft"
model_type="beit-base-patch16-224-pt22k-ft22k"
precision=32
batch_size=128
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

is_tuned="untuned"
strategy="ddp"
preprocess_type="spectogram"
upload_user="MIT"
model_type="ast-finetuned-audioset-10-10-0.4593"
precision=32
batch_size=16
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

is_tuned="untuned"
strategy="ddp"
preprocess_type="spectogram"
upload_user="timm"
model_type="efficientnet_b0.ra_in1k"
precision=32
batch_size=16
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
