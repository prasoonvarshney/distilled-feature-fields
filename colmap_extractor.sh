#!/bin/bash

BASE_PATH="/home/ubuntu/3dvision/project/distilled-feature-fields"
DATASET_DIRECTORY="${BASE_PATH}/data/nerf_llff_data"
DATASET_NAME="flower"

cd ${DATASET_DIRECTORY}
colmap feature_extractor --ImageReader.camera_model OPENCV --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path ${DATASET_NAME}/database.db --image_path ${DATASET_NAME}/images --SiftExtraction.use_gpu=false
colmap exhaustive_matcher --SiftMatching.guided_matching=true --database_path ${DATASET_NAME}/database.db --SiftMatching.use_gpu=false
mkdir ${DATASET_NAME}/sparse
colmap mapper --database_path ${DATASET_NAME}/database.db --image_path ${DATASET_NAME}/images --output_path ${DATASET_NAME}/sparse
colmap bundle_adjuster --input_path ${DATASET_NAME}/sparse/0 --output_path ${DATASET_NAME}/sparse/0 --BundleAdjustment.refine_principal_point 1
colmap image_undistorter --image_path ${DATASET_NAME}/images --input_path ${DATASET_NAME}/sparse/0 --output_path ${DATASET_NAME}_undis --output_type COLMAP
