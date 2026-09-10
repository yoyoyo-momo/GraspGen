#!/usr/bin/env python3
"""Minimal inference script for the trained TriFinger GraspGen model.

Loads the mug.obj, samples a point cloud, runs the diffusion model,
and saves predicted grasp transforms + joint angles to JSON.

Usage:
    python infer_trifinger.py [--checkpoint PATH] [--num_grasps N] [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRASPGEN_ROOT = Path(__file__).parent.parent
CHECKPOINT    = GRASPGEN_ROOT / "runs/trifinger_mug_gen/last.pth"
OBJ_PATH      = Path("/home/j300/GraspDataGen/objects/mug.obj")
GRIPPER_NAME  = "trifinger"
NUM_POINTS    = 1024
NUM_GRASPS    = 114    # must match training num_grasps_per_object


def sample_pointcloud(mesh_path: Path, n_points: int) -> np.ndarray:
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def load_model(checkpoint: Path, gripper_name: str, num_grasps: int):
    sys.path.insert(0, str(GRASPGEN_ROOT / "scripts"))
    from grasp_gen.models.grasp_gen import GraspGenGenerator

    cfg_path = checkpoint.parent / "config.yaml"
    with open(cfg_path) as f:
        import omegaconf
        cfg = omegaconf.OmegaConf.load(cfg_path)

    model = GraspGenGenerator.from_config(cfg.diffusion)
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model, pts: np.ndarray, num_grasps: int, device="cuda"):
    model = model.to(device)
    pts_t = torch.from_numpy(pts).unsqueeze(0).to(device)  # [1, N, 3]
    data = {"points": pts_t}

    # Override num_grasps_per_object temporarily if different from training default
    orig = model.num_grasps_per_object
    model.num_grasps_per_object = num_grasps

    with torch.no_grad():
        outputs, _, stats = model.infer(data, return_metrics=False)

    model.num_grasps_per_object = orig
    return outputs


def filter_colliding_grasps(
    grasps: np.ndarray,
    obj_path: Path,
    min_approach_dot: float = 0.85,
    palm_plate_depth: float = 0.025,
) -> np.ndarray:
    """Return boolean mask of grasps that are collision-free and well-oriented.

    Three checks:
      1. Palm center not inside the object mesh.
      2. A point offset along the approach axis by palm_plate_depth (approximating
         the front face of the palm plate body) is also not inside the mesh.
         This catches cases where the palm center is just outside but the physical
         palm plate still clips through the object.
      3. Gripper approach axis (col 2 of R) points toward the object center
         with dot product >= min_approach_dot (training minimum ~0.85).
    """
    import trimesh

    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    positions = grasps[:, :3, 3]          # [N, 3]
    z_axes    = grasps[:, :3, 2]          # approach direction (col 2)

    # Check 1: palm center inside mesh
    inside = mesh.contains(positions)

    # Check 2: palm-plate front face inside mesh
    # Move palm_plate_depth along the approach direction (toward object) and re-check.
    plate_front = positions + palm_plate_depth * z_axes
    plate_inside = mesh.contains(plate_front)

    # Check 3: approach direction points toward object center
    to_obj = -positions
    norms = np.linalg.norm(to_obj, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    dots = np.sum(z_axes * (to_obj / norms), axis=1)

    valid = ~inside & ~plate_inside & (dots >= min_approach_dot)
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--num_grasps", type=int, default=NUM_GRASPS)
    parser.add_argument("--out", type=Path, default=Path("/tmp/trifinger_predicted_grasps.json"))
    parser.add_argument("--min_approach_dot", type=float, default=0.85,
                        help="Min cosine similarity between gripper Z-axis and vector toward object center (training min ~0.85).")
    parser.add_argument("--palm_plate_depth", type=float, default=0.025,
                        help="Distance (m) along approach axis to offset palm center when checking palm-plate clearance.")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, GRIPPER_NAME, args.num_grasps)

    print(f"Sampling {NUM_POINTS} points from {OBJ_PATH}")
    pts = sample_pointcloud(OBJ_PATH, NUM_POINTS)
    print(f"  Point cloud bounds: {pts.min(axis=0).round(4)} → {pts.max(axis=0).round(4)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {device} ({args.num_grasps} grasps)...")
    outputs = run_inference(model, pts, args.num_grasps, device=device)

    grasps  = outputs["grasps_pred"][0].cpu().numpy()  # [N, 4, 4]
    q_pre   = outputs["q_pre"][0].cpu().numpy()        # [N, 8]
    q_final = outputs["q_final"][0].cpu().numpy()      # [N, 8]

    print(f"Generated {grasps.shape[0]} grasps")

    # Filter out grasps that collide with or point away from the object
    valid_mask = filter_colliding_grasps(grasps, OBJ_PATH, args.min_approach_dot, args.palm_plate_depth)
    n_removed = (~valid_mask).sum()
    if n_removed:
        print(f"  Filtered {n_removed} colliding/mis-oriented grasps → {valid_mask.sum()} remain")
    grasps  = grasps[valid_mask]
    q_pre   = q_pre[valid_mask]
    q_final = q_final[valid_mask]

    print(f"  Translation range (x): {grasps[:,0,3].min():.4f} → {grasps[:,0,3].max():.4f}")
    print(f"  Translation range (y): {grasps[:,1,3].min():.4f} → {grasps[:,1,3].max():.4f}")
    print(f"  Translation range (z): {grasps[:,2,3].min():.4f} → {grasps[:,2,3].max():.4f}")
    print(f"  q_pre  range: {q_pre.min():.4f} → {q_pre.max():.4f}")
    print(f"  q_final range: {q_final.min():.4f} → {q_final.max():.4f}")

    # Joint names in canonical order (trifinger.yaml)
    joint_names = ["A_1_joint","A_2_joint","A_3_joint",
                   "B_1_joint","B_2_joint","B_3_joint",
                   "C_2_joint","C_3_joint"]

    out = {
        "model": str(args.checkpoint),
        "object": "objects/mug.obj",
        "num_grasps": int(grasps.shape[0]),
        "filter": {"min_approach_dot": args.min_approach_dot,
                   "palm_plate_depth": args.palm_plate_depth,
                   "removed": int(n_removed)},
        "grasps": [],
    }

    for i in range(grasps.shape[0]):
        T = grasps[i]
        entry = {
            "id": i,
            "transform": T.tolist(),
            "position": T[:3, 3].tolist(),
            "q_pre":   {j: float(q_pre[i, k])   for k, j in enumerate(joint_names)},
            "q_final": {j: float(q_final[i, k]) for k, j in enumerate(joint_names)},
        }
        out["grasps"].append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(out['grasps'])} grasps → {args.out}")


if __name__ == "__main__":
    main()
