#!/usr/bin/env bash
# Retrain GraspGen generator on the combined self-collision-checked dataset.
#
# mug_handle_combined_final.yaml = 658 (narrow/flanking A_1-B_1 spread) +
# 441 (widened/free A_1-B_1 spread, self-collision filtered) = 1099 grasps.
# First dataset in the project where every grasp passed: correct
# q_final target application, displacement success metric, 1.5 N*m torque,
# penetration filter, AND gripper self-collision filter.
#
# Usage:
#   bash runs/train_graspgen_trifinger_1099.sh [--resume]
#     --resume  continue from last.pth instead of starting fresh

set -e

GRIPPER_NAME="trifinger"
DATASET_DIR="/home/j300/GraspDataGen"
COMBINED_YAML="${DATASET_DIR}/grasp_sim_data/ThreeFigV2_USDZ_1_a_0.01_black_blue_J002_mod2_fixed/mug_handle_combined_final.yaml"
TRAINING_JSON="${DATASET_DIR}/grasp_training_data/ThreeFigV2_USDZ/mug.json"
OBJECT_PATH="${DATASET_DIR}/objects/mug.obj"

PYTHON=/home/j300/ITRI-GraspGen/.venv/bin/python
LOG_DIR="/home/j300/GraspGen/runs/trifinger_mug_handle_gen_1099"
CHECKPOINT="$LOG_DIR/last.pth"
CACHE_DIR="/tmp/graspgen_train_cache_trifinger_1099"

RESUME=false
for arg in "$@"; do
    [ "$arg" = "--resume" ] && RESUME=true
done

# ── Step 1: count grasps ──────────────────────────────────────────────────────
echo "=== Step 1: Counting grasps in ${COMBINED_YAML} ==="
if [ ! -f "${COMBINED_YAML}" ]; then
    echo "ERROR: ${COMBINED_YAML} not found."
    exit 1
fi

NUM_GRASPS=$($PYTHON -c "
import yaml
with open('${COMBINED_YAML}') as f:
    d = yaml.safe_load(f)
print(len(d.get('grasps', {})))
")
echo "  Found ${NUM_GRASPS} grasps"

# ── Step 2: YAML → training JSON ──────────────────────────────────────────────
echo
echo "=== Step 2: Converting YAML → training JSON ==="
cd "${DATASET_DIR}"
$PYTHON scripts/graspgen/yaml_to_training_json.py \
    --input  "${COMBINED_YAML}" \
    --output "${TRAINING_JSON}" \
    --object_path "${OBJECT_PATH}"
echo "  → ${TRAINING_JSON}"

# ── Step 3: Train ─────────────────────────────────────────────────────────────
echo
echo "=== Step 3: Training (num_grasps_per_object=${NUM_GRASPS}) ==="
echo "  Log dir : ${LOG_DIR}"
echo "  Resume  : ${RESUME}"
echo

if [ "$RESUME" = false ]; then
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR"

NGPU=1
NWORKER=4
NEPOCH=2000
BATCH=4
PRINT_FREQ=25
PLOT_FREQ=50
SAVE_FREQ=50
TIMESTEPS=100
BACKBONE="pointnet"
NUM_POINTS=1024
ROTATION_REPR="r3_6d"

cd /home/j300/GraspGen/scripts && $PYTHON train_graspgen.py \
    data.num_points=$NUM_POINTS \
    data.num_object_points=1024 \
    data.dataset_cls="TriFingerGraspDataset" \
    data.rotation_augmentation=True \
    data.cache_dir="$CACHE_DIR" \
    data.root_dir="${DATASET_DIR}/grasp_training_data/ThreeFigV2_USDZ" \
    data.object_root_dir="${DATASET_DIR}" \
    data.grasp_root_dir="${DATASET_DIR}/grasp_training_data/ThreeFigV2_USDZ" \
    data.dataset_name="$GRIPPER_NAME" \
    data.dataset_version="v2" \
    data.prob_point_cloud=-1 \
    data.redundancy=1 \
    data.gripper_name="$GRIPPER_NAME" \
    data.num_grasps_per_object=$NUM_GRASPS \
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
    diffusion.num_grasps_per_object=$NUM_GRASPS \
    2>&1 | tee "$LOG_DIR/console_log.txt"
