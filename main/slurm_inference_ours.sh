#!/usr/bin/env bash
set -x

PARTITION=Zoetrope

INPUT_BASE=$1
CKPT=smpler_x_h32

GPUS=1
JOB_NAME=inference_${INPUT_BASE}

GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=4 # ${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}

IMG_PATH=${INPUT_BASE}/color
SAVE_DIR=${INPUT_BASE}/smplerx

# video to images
mkdir $SAVE_DIR

end_count=$(find "$IMG_PATH" -type f | wc -l)
echo $end_count

python inference.py \
    --num_gpus ${GPUS_PER_NODE} \
    --exp_name output/demo_${JOB_NAME} \
    --pretrained_model ${CKPT} \
    --agora_benchmark agora_model \
    --img_path ${IMG_PATH} \
    --start 1 \
    --end  $end_count \
    --output_folder ${SAVE_DIR} \
    --save_mesh \
    --show_img_side
    # --show_bbox
    # --show_verts \
    # --multi_person \
    # --iou_thr 0.2 \
    # --bbox_thr 20 \



