#!/usr/bin/env python3
"""Inference for TriFinger GraspGen with the root POSE held fixed.

The model's own pose (position + orientation) predictions were measured to
be close to uncorrelated with the correct approach direction (median ~74
degrees off horizontal vs. training data's ~12 degrees), while the training
data itself always uses an almost-constant pose (pinned horizontal approach
onto the handle, only small jitter + free roll). Rather than trust the
model's pose sampling, this script supplies a batch of KNOWN-GOOD target
poses (reused from the training set) and only lets the reverse diffusion
process denoise the joint-angle (q_pre/q_final) sub-vector.

Mechanism: standard diffusion "inpainting"/conditioning. At every reverse
step, the position and rotation sub-vectors are overwritten with the target
pose re-noised to that step's noise level (via the same DDPMScheduler's
forward process, noise_scheduler_{pos,rot}.add_noise) instead of the
network's own reverse prediction for those dims — so the network always
sees a correctly-noised version of the TRUE pose as context, and only its
prediction for the joint-angle dims is kept.

Usage:
    python infer_trifinger_fixed_pose.py \
        --checkpoint runs/trifinger_mug_handle_gen_6047_norot/last.pth \
        --poses_from grasp_sim_data/.../mug_handle_combined_6047.yaml \
        --num_poses 300 \
        --out /tmp/predicted_fixed_pose.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml

GRASPGEN_ROOT = Path(__file__).parent.parent
OBJ_PATH = Path("/home/j300/GraspDataGen/objects/mug.obj")
GRIPPER_NAME = "trifinger"
NUM_POINTS = 1024


def quat_mat(w, x, y, z):
    import math
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def sample_pointcloud(mesh_path, n_points):
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def load_target_poses(poses_from, num_poses, seed):
    with open(poses_from) as f:
        data = yaml.load(f, Loader=yaml.CSafeLoader)
    grasps = list(data["grasps"].values())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(grasps), size=min(num_poses, len(grasps)), replace=False)
    mats = []
    for i in idx:
        g = grasps[i]
        pos = np.array(g["position"][:3] if isinstance(g["position"], list) else
                        [g["position"]["x"], g["position"]["y"], g["position"]["z"]])
        o = g["orientation"]
        R = quat_mat(o["w"], *o["xyz"])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        mats.append(T)
    return np.array(mats, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--poses_from", type=Path, required=True,
                         help="YAML to draw known-good target poses from (position+orientation reused as-is).")
    parser.add_argument("--num_poses", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("/tmp/trifinger_predicted_fixed_pose.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(GRASPGEN_ROOT / "scripts"))
    from grasp_gen.models.grasp_gen import GraspGenGenerator
    from grasp_gen.models.model_utils import offset2batch
    from grasp_gen.utils.math_utils import matrix_to_rt, rt_to_matrix
    import omegaconf

    cfg_path = args.checkpoint.parent / "config.yaml"
    cfg = omegaconf.OmegaConf.load(cfg_path)
    model = GraspGenGenerator.from_config(cfg.diffusion)
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Sampling {NUM_POINTS} points from {OBJ_PATH}")
    pts = sample_pointcloud(OBJ_PATH, NUM_POINTS)
    pts_t = torch.from_numpy(pts).unsqueeze(0).to(device)  # [1, N, 3]

    print(f"Loading {args.num_poses} target poses from {args.poses_from}")
    target_mats = load_target_poses(args.poses_from, args.num_poses, args.seed)
    num_grasps = target_mats.shape[0]
    print(f"  -> {num_grasps} target poses")

    target_T = torch.from_numpy(target_mats).float().to(device)  # [N,4,4]
    target_rt = matrix_to_rt(target_T, model.grasp_repr, model.kappa)  # [N, pose_dim]

    # ---- replicate forward_inference, but with pose held fixed ----
    with torch.no_grad():
        depth = pts_t.reshape([-1, NUM_POINTS, 3]).to(device)
        if model.kappa is not None:
            depth = model.kappa * depth
        object_embedding = model.object_encoder(depth)  # [1, num_obs_dim]
        offset = torch.tensor([num_grasps]).to(device)
        mask_batch = offset2batch(offset)
        object_embedding = object_embedding[mask_batch]  # [num_grasps, num_obs_dim]

        batch_size = num_grasps
        noisy_grasps = torch.randn([batch_size, model.output_dim], device=device)

        model.noise_scheduler_pos.set_timesteps(model.num_diffusion_iters_eval)
        timesteps = model.noise_scheduler_pos.timesteps
        model.noise_scheduler_rot.set_timesteps(model.num_diffusion_iters_eval)

        clean_pos = target_rt[:, :3]       # [N,3]
        clean_rot = target_rt[:, 3:9]      # [N,6]

        for k in timesteps:
            samples = noisy_grasps
            noise_pred = model.diffusion_head(object_embedding, k, samples)

            res_pos = model.noise_scheduler_pos.step(
                model_output=noise_pred[..., :3], timestep=k, sample=noisy_grasps[..., :3])
            res_rot = model.noise_scheduler_rot.step(
                model_output=noise_pred[..., 3:], timestep=k, sample=noisy_grasps[..., 3:])

            joints_part = res_rot.prev_sample[:, 6:]  # network's own reverse step, q_pre+q_final

            if int(k) > 0:
                forced_pos = model.noise_scheduler_pos.add_noise(
                    clean_pos, torch.randn_like(clean_pos), k)
                forced_rot = model.noise_scheduler_rot.add_noise(
                    clean_rot, torch.randn_like(clean_rot), k)
            else:
                forced_pos = clean_pos
                forced_rot = clean_rot

            noisy_grasps = torch.cat([forced_pos, forced_rot, joints_part], dim=-1)

        # Final: hard-set pose to the exact known target (no residual noise).
        noisy_grasps = torch.cat([clean_pos, clean_rot, noisy_grasps[:, 9:]], dim=-1)

        T_out = rt_to_matrix(noisy_grasps[:, :model.pose_dim], model.grasp_repr, model.kappa)
        q_pre_out = noisy_grasps[:, model.pose_dim: model.pose_dim + model.num_joints]
        q_final_out = noisy_grasps[:, model.pose_dim + model.num_joints: model.output_dim]
        # noisy_grasps' joint sub-vector is normalized to [-1, 1]; denormalize to radians
        # (see generator.py forward_inference, which calls this same method).
        q_pre_out = model._denormalize_joints(q_pre_out)
        q_final_out = model._denormalize_joints(q_final_out)

    grasps_np = T_out.cpu().numpy()
    q_pre_np = q_pre_out.cpu().numpy()
    q_final_np = q_final_out.cpu().numpy()

    joint_names = ["A_1_joint", "A_2_joint", "A_3_joint",
                   "B_1_joint", "B_2_joint", "B_3_joint",
                   "C_2_joint", "C_3_joint"]

    out = {
        "model": str(args.checkpoint),
        "object": "objects/mug.obj",
        "num_grasps": int(num_grasps),
        "grasps": [],
    }
    for i in range(num_grasps):
        T = grasps_np[i]
        entry = {
            "id": i,
            "transform": T.tolist(),
            "position": T[:3, 3].tolist(),
            "q_pre":   {j: float(q_pre_np[i, k]) for k, j in enumerate(joint_names)},
            "q_final": {j: float(q_final_np[i, k]) for k, j in enumerate(joint_names)},
        }
        out["grasps"].append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(out['grasps'])} grasps (fixed pose, model-inferred joints) -> {args.out}")


if __name__ == "__main__":
    main()
