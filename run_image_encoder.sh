#!/bin/bash

BASE_PATH="/home/ubuntu/3dvision/project/distilled-feature-fields"
cd "${BASE_PATH}/encoders/lseg_encoder/"

DATASET_DIRECTORY="${BASE_PATH}/data/nerf_llff_data/room"
python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir "${DATASET_DIRECTORY}/rgb_feature_langseg" --test-rgb-dir "${DATASET_DIRECTORY}/images"

DATASET_DIRECTORY="${BASE_PATH}/data/nerf_llff_data/trex"
python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir "${DATASET_DIRECTORY}/rgb_feature_langseg" --test-rgb-dir "${DATASET_DIRECTORY}/images"

DATASET_DIRECTORY="${BASE_PATH}/data/nerf_llff_data/orchids"
python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir "${DATASET_DIRECTORY}/rgb_feature_langseg" --test-rgb-dir "${DATASET_DIRECTORY}/images"

DATASET_DIRECTORY="${BASE_PATH}/data/nerf_llff_data/horns"
python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir "${DATASET_DIRECTORY}/rgb_feature_langseg" --test-rgb-dir "${DATASET_DIRECTORY}/images"

DATASET_DIRECTORY="${BASE_PATH}/data/nerf_llff_data/leaves"
python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir "${DATASET_DIRECTORY}/rgb_feature_langseg" --test-rgb-dir "${DATASET_DIRECTORY}/images"


cd "${BASE_PATH}"