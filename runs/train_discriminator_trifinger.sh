#!/usr/bin/env bash
# Train a GraspGenDiscriminator to score TriFinger grasp proposals on the mug.
#
# IMPORTANT LIMITATION (found via code research 2026-09-04): GraspGenDiscriminator
# only ever scores the 4x4 root pose — it never sees q_pre/q_final (finger joint
# angles). It can learn to reject badly-placed root poses (real problem: our
# generator misses the handle by ~54mm median) but CANNOT learn to reject a
# good root pose paired with a wall-penetrating finger configuration (our other
# real problem: 78% of the 1099-model's own "successes" were penetration
# artifacts). Treat this as a partial mitigation, not a full fix.
#
# Data: built from a grasp_sim.py run with --save_fails, then normalized with
# prep_discriminator_data.py (copies pregrasp_position/orientation over the
# final position/orientation fields so positive and negative examples have
# consistent provenance — see that script's docstring for why this matters),
# then converted with the same yaml_to_training_json.py used for the generator.
#
# KNOWN CONFIG BUG (found via code research 2026-09-04): scripts/config.yaml's
# discriminator_ratio default has 5 elements, but
# load_discriminator_batch_with_stratified_sampling (grasp_gen/dataset/dataset.py)
# unconditionally indexes positions 5 and 6 (pos_true_onpolicy/neg_true_onpolicy)
# -> IndexError on the first batch with the stock config. Padded to 7 below.
#
# Usage:
#   bash runs/train_discriminator_trifinger.sh <discriminator_ready.yaml> [--resume]

set -e

GRIPPER_NAME="trifinger"
DATASET_DIR="/home/j300/GraspDataGen"
INPUT_YAML="$1"
shift || true
RESUME=false
for arg in "$@"; do
    [ "$arg" = "--resume" ] && RESUME=true
done

if [ -z "$INPUT_YAML" ] || [ ! -f "$INPUT_YAML" ]; then
    echo "ERROR: pass a discriminator-ready (prep_discriminator_data.py output) YAML as \$1"
    exit 1
fi

TRAINING_JSON="${DATASET_DIR}/grasp_training_data/ThreeFigV2_USDZ_discriminator/mug.json"
OBJECT_PATH="${DATASET_DIR}/objects/mug.obj"

PYTHON=/home/j300/ITRI-GraspGen/.venv/bin/python
LOG_DIR="/home/j300/GraspGen/runs/trifinger_mug_handle_discriminator"
CHECKPOINT="$LOG_DIR/last.pth"
CACHE_DIR="/tmp/graspgen_train_cache_trifinger_discriminator"

echo "=== Step 1: Converting YAML -> training JSON ==="
cd "${DATASET_DIR}"
$PYTHON scripts/graspgen/yaml_to_training_json.py \
    --input  "${INPUT_YAML}" \
    --output "${TRAINING_JSON}" \
    --object_path "${OBJECT_PATH}"
echo "  -> ${TRAINING_JSON}"

echo
echo "=== Step 2: Training discriminator ==="
echo "  Log dir : ${LOG_DIR}"
echo "  Resume  : ${RESUME}"

if [ "$RESUME" = false ]; then
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR"

cd /home/j300/GraspGen/scripts && $PYTHON train_graspgen.py \
    data.num_points=1024 \
    data.num_object_points=1024 \
    data.dataset_cls="TriFingerGraspDataset" \
    data.rotation_augmentation=True \
    data.cache_dir="$CACHE_DIR" \
    data.root_dir="${DATASET_DIR}/grasp_training_data/ThreeFigV2_USDZ_discriminator" \
    data.object_root_dir="${DATASET_DIR}" \
    data.grasp_root_dir="${DATASET_DIR}/grasp_training_data/ThreeFigV2_USDZ_discriminator" \
    data.dataset_name="$GRIPPER_NAME" \
    data.dataset_version="v2" \
    data.gripper_name="$GRIPPER_NAME" \
    data.num_grasps_per_object=128 \
    data.load_discriminator_dataset=True \
    'data.discriminator_ratio=[0.50,0.20,0.25,0.05,0.0,0.0,0.0]' \
    train.log_dir="$LOG_DIR" \
    train.batch_size=4 \
    train.num_gpus=1 \
    train.num_epochs=500 \
    train.num_workers=4 \
    train.print_freq=25 \
    train.plot_freq=50 \
    train.save_freq=50 \
    train.checkpoint="$CHECKPOINT" \
    train.model_name="discriminator" \
    optimizer.type="ADAMW" \
    optimizer.lr=0.00005 \
    optimizer.grad_clip=-1 \
    discriminator.gripper_name="$GRIPPER_NAME" \
    discriminator.obs_backbone="pointnet" \
    discriminator.grasp_repr="r3_6d" \
    discriminator.pose_repr="mlp" \
    2>&1 | tee "$LOG_DIR/console_log.txt"
