#!/usr/bin/env bash
# Train GraspGen generator (diffusion) for ThreeFigV2 trifinger on mug object.
#
# Data:  114 unique successful grasps from Isaac Sim (seeds 0, 42, 100-119).
# Model: Diffusion generator with PointNet++ encoder, 25D output
#        (SE(3) pose + 8 pregrasp joints + 8 close joints).
#
# Usage:  bash runs/train_graspgen_trifinger_gen.sh [--resume]
#   --resume  load last.pth from LOG_DIR if it exists (default: start fresh)

set -e

GRIPPER_NAME="trifinger"
NGPU=1
NWORKER=4
NEPOCH=2000
BATCH=4                  # small batch — only 1 object in the dataset
PRINT_FREQ=25
PLOT_FREQ=50
SAVE_FREQ=50
TIMESTEPS=100
NUM_GRASPS_PER_OBJ=114  # all available grasps
BACKBONE="pointnet"      # ptv3 needs extra install; pointnet is lighter
NUM_POINTS=1024          # num scene points (object only, no scene context)
ROTATION_REPR="r3_6d"

OBJECT_DATASET_DIR="/home/j300/GraspDataGen"
GRASP_DATASET_DIR="/home/j300/GraspDataGen/grasp_training_data/ThreeFigV2_USDZ"
SPLIT_DATASET_DIR="/home/j300/GraspDataGen/grasp_training_data/ThreeFigV2_USDZ"
CACHE_DIR="/tmp/graspgen_train_cache_trifinger"
LOG_DIR="/home/j300/GraspGen/runs/trifinger_mug_gen"
CHECKPOINT="$LOG_DIR/last.pth"

RESUME=false
for arg in "$@"; do
    [ "$arg" = "--resume" ] && RESUME=true
done

if [ "$RESUME" = false ]; then
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR"

echo "=== GraspGen Trifinger Generator Training ==="
echo "Log dir : $LOG_DIR"
echo "Resume  : $RESUME"
echo

PYTHON=/home/j300/ITRI-GraspGen/.venv/bin/python
cd /home/j300/GraspGen/scripts && $PYTHON train_graspgen.py \
    data.num_points=$NUM_POINTS \
    data.num_object_points=1024 \
    data.dataset_cls="TriFingerGraspDataset" \
    data.rotation_augmentation=True \
    data.cache_dir="$CACHE_DIR" \
    data.root_dir="$SPLIT_DATASET_DIR" \
    data.object_root_dir="$OBJECT_DATASET_DIR" \
    data.grasp_root_dir="$GRASP_DATASET_DIR" \
    data.dataset_name="$GRIPPER_NAME" \
    data.dataset_version="v2" \
    data.prob_point_cloud=-1 \
    data.redundancy=1 \
    data.gripper_name="$GRIPPER_NAME" \
    data.num_grasps_per_object=$NUM_GRASPS_PER_OBJ \
    data.load_discriminator_dataset=False \
    data.load_contact=False \
    data.visualize_batch=False \
    train.log_dir="$LOG_DIR" \
    train.batch_size=$BATCH \
    train.num_gpus=$NGPU \
    train.num_epochs=$NEPOCH \
    train.num_workers=$NWORKER \
    train.print_freq=$PRINT_FREQ \
    train.plot_freq=$PLOT_FREQ \
    train.save_freq=$SAVE_FREQ \
    train.checkpoint="$CHECKPOINT" \
    train.model_name="diffusion" \
    optimizer.type="ADAMW" \
    optimizer.lr=0.00005 \
    optimizer.grad_clip=-1 \
    diffusion.gripper_name="$GRIPPER_NAME" \
    diffusion.num_diffusion_iters=$TIMESTEPS \
    diffusion.num_diffusion_iters_eval=$TIMESTEPS \
    diffusion.obs_backbone="$BACKBONE" \
    diffusion.grasp_repr="$ROTATION_REPR" \
    diffusion.attention="cat" \
    diffusion.compositional_schedular=True \
    diffusion.loss_pointmatching=True \
    diffusion.loss_l1_pos=False \
    diffusion.loss_l1_rot=False \
    diffusion.loss_l1_joints=True \
    diffusion.pose_repr="mlp" \
    diffusion.num_grasps_per_object=$NUM_GRASPS_PER_OBJ \
    2>&1 | tee "$LOG_DIR/console_log.txt"
