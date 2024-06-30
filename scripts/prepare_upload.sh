#!/bin/bash

path="src/postprocessing"
is_tuned="untuned"
strategy="ddp"
preprocess_type="spectogram"
upload_user="MIT"
model_type="ast-finetuned-audioset-10-10-0.4593"
precision=32
batch_size=16
epoch=10

python $path/prepare_upload.py \
    is_tuned=$is_tuned \
    strategy=$strategy \
    preprocess_type=$preprocess_type \
    upload_user=$upload_user \
    model_type=$model_type \
    precision=$precision \
    batch_size=$batch_size \
    epoch=$epoch
