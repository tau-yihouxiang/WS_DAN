#!/usr/bin/env bash

DATASET="Bird"
TRAIN_DIR="./$DATASET/WS_DAN/TRAIN/ws_dan_part_32"
TEST_DIR="./$DATASET/WS_DAN/TEST/ws_dan_part_32"

python eval_sample.py --checkpoint_path=$TRAIN_DIR \
                         --dataset_name=$DATASET \
                         --dataset_split_name='test' \
                         --dataset_dir="./$DATASET/Data/tfrecords" \
                         --eval_dir=$TEST_DIR \
                         --model_name='inception_v3_bap' \
                         --batch_size=16 \
                         --eval_image_size=448\
                         --gpus="1"\
                         --num_classes=200\
                         --feature_maps="Mixed_6e"\
                         --attention_maps="Mixed_7a_b0"\
                         --num_parts=32
