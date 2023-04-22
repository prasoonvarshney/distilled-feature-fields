#!/bin/bash

BASE_PATH="/home/ubuntu/3dvision/project/distilled-feature-fields"
DATASET_DIRECTORY="${BASE_PATH}/data/nerf_llff_data/flower"
EXPERIMENT_NAME="exp_flower_v3"
DATASET_NAME="colmap"
CHECKPOINTS_DIRECTORY="${BASE_PATH}/ckpts/${DATASET_NAME}/${EXPERIMENT_NAME}_clip"
CLIPNERF_TEXT="purple_flower"
# Dataset 1: Apples bananas etc. (sample_dataset)

# # Step 1: From images to LSeg features
# cd "${BASE_PATH}/encoders/lseg_encoder/"
# python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir "${DATASET_DIRECTORY}/rgb_feature_langseg" --test-rgb-dir "${DATASET_DIRECTORY}/images"
# cd "${BASE_PATH}"
# # Step 2: Train DFF
# python train.py --root_dir ${DATASET_DIRECTORY} --dataset_name ${DATASET_NAME} --exp_name ${EXPERIMENT_NAME} --downsample 0.25 --num_epochs 4 --batch_size 4096 --scale 4.0 --ray_sampling_strategy same_image --feature_dim 512 --random_bg --feature_directory "${DATASET_DIRECTORY}/rgb_feature_langseg"
# Step 3: Optimize CLIP-NERF
rm -rf ${CHECKPOINTS_DIRECTORY}
python train.py --root_dir ${DATASET_DIRECTORY} --dataset_name ${DATASET_NAME} --exp_name ${EXPERIMENT_NAME}_clip --downsample 0.25 --num_epochs 1 --batch_size 4096 --scale 4.0 --ray_sampling_strategy same_image --feature_dim 512 --random_bg --clipnerf_text ${CLIPNERF_TEXT} --clipnerf_filter_text flower leaves ground --weight_path ckpts/${DATASET_NAME}/${EXPERIMENT_NAME}/epoch=3_slim.ckpt --accumulate_grad_batches 2
# # # Step 4: Render with edits
python render.py --root_dir ${DATASET_DIRECTORY} --dataset_name ${DATASET_NAME} --exp_name ${EXPERIMENT_NAME}_clip --downsample 0.25 --scale 4.0 --ray_sampling_strategy same_image --feature_dim 512 --ckpt_path ckpts/${DATASET_NAME}/${EXPERIMENT_NAME}_clip/epoch=0_slim.ckpt --clipnerf_text ${CLIPNERF_TEXT} --edit_config query.yaml
