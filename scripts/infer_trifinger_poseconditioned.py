#!/usr/bin/env python3
"""Inference for the pose-conditioned TriFinger GraspGen model.

Unlike infer_trifinger.py (which samples pose AND joints from pure noise)
or infer_trifinger_fixed_pose.py (a repaint-style hack that holds pose fixed
during a model that was still trained to diffuse it), this calls a model
trained with condition_on_pose=True — pose is a genuine conditioning input
to that model's forward_inference, not something bolted on at inference
time. Target poses are generated FRESH here using the same geometric recipe
gen_human_grab_poses.py uses to build the training data (point on the
handle arc + standoff + horizontal approach with jitter + free roll) — this
is a deployment-realistic test, not a replay of known training poses.

Usage:
    python infer_trifinger_poseconditioned.py \
        --checkpoint runs/trifinger_mug_handle_gen_poseconditioned/last.pth \
        --num_poses 500 --seed 0 \
        --out /tmp/predicted_poseconditioned.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

GRASPGEN_ROOT = Path(__file__).parent.parent
OBJ_PATH = Path("/home/j300/GraspDataGen/objects/mug.obj")
NUM_POINTS = 1024

# Same constants as gen_sideways_handle_poses.py / gen_human_grab_poses.py
BASE_LENGTH = 0.0858
HANDLE_Y_HALF = 0.020
DIR_JITTER = 0.14


def rot_from_z_and_roll(z_axis, roll):
    z = z_axis / np.linalg.norm(z_axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = math.cos(roll), math.sin(roll)
    xr = c * x + s * y
    R = np.column_stack([xr, np.cross(z, xr), z])
    return R


def generate_fresh_poses(obj_path, num_poses, seed, min_standoff=0.005, max_standoff=0.030):
    """Same recipe as gen_human_grab_poses.py: point on the outer handle arc,
    horizontal approach (+jitter), random standoff, fully free roll."""
    rng = np.random.default_rng(seed)
    obj = trimesh.load(str(obj_path), force="mesh", process=False)
    surf, _ = trimesh.sample.sample_surface(obj, 30000)
    surf = np.asarray(surf)
    hmask = (surf[:, 0] > 0.045) & (np.abs(surf[:, 1]) < HANDLE_Y_HALF)
    handle_pts = surf[hmask]
    print(f"Outer-arc handle points: {len(handle_pts)}")

    mats = []
    for _ in range(num_poses):
        target = handle_pts[rng.integers(len(handle_pts))]
        standoff = rng.uniform(min_standoff, max_standoff)
        d = np.array([-1.0, 0.0, 0.0])
        d += rng.normal(0, DIR_JITTER, 3) * np.array([0.3, 1.0, 1.0])
        d /= np.linalg.norm(d)
        root = target - (BASE_LENGTH + standoff) * d
        roll = rng.uniform(0, 2 * math.pi)
        R = rot_from_z_and_roll(d, roll)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = root
        mats.append(T)
    return np.array(mats, dtype=np.float32)


def sample_pointcloud(mesh_path, n_points):
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_poses", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("/tmp/trifinger_predicted_poseconditioned.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(GRASPGEN_ROOT / "scripts"))
    from grasp_gen.models.generator import GraspGenGenerator
    import omegaconf

    cfg_path = args.checkpoint.parent / "config.yaml"
    cfg = omegaconf.OmegaConf.load(cfg_path)
    model = GraspGenGenerator.from_config(cfg.diffusion)
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    assert model.condition_on_pose, "checkpoint's config.yaml does not have condition_on_pose=True"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Sampling {NUM_POINTS} points from {OBJ_PATH}")
    pts = sample_pointcloud(OBJ_PATH, NUM_POINTS)
    pts_t = torch.from_numpy(pts).unsqueeze(0).to(device)  # [1, N, 3]

    print(f"Generating {args.num_poses} fresh target poses (handle-arc recipe, seed={args.seed})")
    target_mats = generate_fresh_poses(OBJ_PATH, args.num_poses, args.seed)
    target_T = torch.from_numpy(target_mats).float().to(device).unsqueeze(0)  # [1, N, 4, 4]

    data = {"points": pts_t, "grasps": [target_T[0]]}
    with torch.no_grad():
        outputs, _, _ = model.infer(data, return_metrics=False)

    T_palm = outputs["T_palm"][0].cpu().numpy()       # [N,4,4]
    q_pre = outputs["q_pre"][0].cpu().numpy()         # [N,8]
    q_final = outputs["q_final"][0].cpu().numpy()     # [N,8]

    joint_names = ["A_1_joint", "A_2_joint", "A_3_joint",
                   "B_1_joint", "B_2_joint", "B_3_joint",
                   "C_2_joint", "C_3_joint"]

    out = {
        "model": str(args.checkpoint),
        "object": "objects/mug.obj",
        "num_grasps": int(T_palm.shape[0]),
        "grasps": [],
    }
    for i in range(T_palm.shape[0]):
        T = T_palm[i]
        entry = {
            "id": i,
            "transform": T.tolist(),
            "position": T[:3, 3].tolist(),
            "q_pre":   {j: float(q_pre[i, k]) for k, j in enumerate(joint_names)},
            "q_final": {j: float(q_final[i, k]) for k, j in enumerate(joint_names)},
        }
        out["grasps"].append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(out['grasps'])} grasps (fresh pose, model-inferred joints) -> {args.out}")


if __name__ == "__main__":
    main()
