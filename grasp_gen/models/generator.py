#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from scipy.spatial import KDTree

from grasp_gen.utils.math_utils import (
    matrix_to_rt,
    rt_to_matrix,
    components_to_grasp_state,
)
from grasp_gen.metrics import compute_metrics_given_two_sets_of_poses, compute_recall
from grasp_gen.models.model_utils import (
    ContactHeatmapHead,
    PointNetPlusPlus,
    SinusoidalPosEmb,
    compute_grasp_loss,
    convert_to_ptv3_pc_format,
    focal_loss_with_logits,
    load_pretrained_checkpoint_to_dict,
    offset2batch,
)
# NOTE: PointTransformerV3 is imported lazily where used to avoid forcing the
# spconv dependency when the pointnet backbone (the default) is used instead.
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


class GraspGenGenerator(nn.Module):
    """GraspGen generator model for generating 6-DOF robotic grasps.

    This class implements a diffusion model that generates robotic grasping poses from point cloud observations.
    It uses a combination of object encoding and diffusion-based denoising to generate high-quality grasps.

    Args:
        num_embed_dim (int): Dimension of embedding vectors. Default: 256
        num_obs_dim (int): Dimension of observation features. Default: 512
        diffusion_embed_dim (int): Dimension of diffusion step embeddings. Default: 512
        image_size (int): Size of input images if using vision backbone. Default: 256
        num_diffusion_iters (int): Number of diffusion steps for training. Default: 100
        num_diffusion_iters_eval (int): Number of diffusion steps for evaluation. Default: 100
        obs_backbone (str): Type of observation encoder backbone ('vit', 'pointnet', 'ptv3'). Default: 'vit'
        compositional_schedular (bool): Whether to use separate schedulers for position and rotation. Default: False
        loss_pointmatching (bool): Whether to use point matching loss. Default: True
        loss_l1_pos (bool): Whether to use L1 loss on positions. Default: False
        loss_l1_rot (bool): Whether to use L1 loss on rotations. Default: False
        grasp_repr (str): Grasp representation type ('r3_6d', 'r3_so3', 'r3_euler'). Default: 'r3_6d'
        kappa (float): Scale factor for noise. Default: -1.0
        clip_sample (bool): Whether to clip samples in diffusion process. Default: True
        beta_schedule (str): Schedule for noise variance. Default: 'beta_schedule'
        attention (str): Type of attention mechanism. Default: 'cat'
        grid_size (float): Grid size for point cloud processing. Default: 0.02
        gripper_name (str): Name of the gripper model. Default: 'franka_panda'
        pose_repr (str): Type of pose representation. Default: 'mlp'
        num_grasps_per_object (int): Number of grasps to generate per object. Default: 20
        checkpoint_object_encoder_pretrained (str): Path to pretrained object encoder. Default: None
        num_joints (int): Number of finger joints per grasp state (q_pre + q_final = 2*num_joints). Default: 8
        num_fingers (int): Number of gripper fingers; sets contact heatmap output channels. Default: 3
    """

    def __init__(
        self,
        num_embed_dim: int = 256,
        num_obs_dim: int = 512,
        diffusion_embed_dim: int = 512,
        image_size: int = 256,
        num_diffusion_iters: int = 100,
        num_diffusion_iters_eval: int = 100,
        obs_backbone: str = "pointnet",
        compositional_schedular: bool = False,
        loss_pointmatching: bool = True,
        loss_l1_pos: bool = False,
        loss_l1_rot: bool = False,
        loss_l1_joints: bool = False,
        grasp_repr: str = "r3_6d",
        kappa: float = -1.0,
        clip_sample: bool = True,
        beta_schedule: str = "beta_schedule",
        attention: str = "cat",
        grid_size: float = 0.02,
        gripper_name: str = "franka_panda",
        pose_repr: str = "mlp",
        num_grasps_per_object: int = 20,
        checkpoint_object_encoder_pretrained: str = None,
        num_joints: int = 8,
        num_fingers: int = 3,
        condition_on_pose: bool = False,
    ):
        super().__init__()

        self.num_embed_dim = num_embed_dim
        self.num_obs_dim = num_obs_dim
        self.diffusion_embed_dim = diffusion_embed_dim
        self.image_size = image_size
        self.num_diffusion_iters = num_diffusion_iters
        self.num_diffusion_iters_eval = num_diffusion_iters_eval
        self.obs_backbone = obs_backbone
        self.compositional_schedular = compositional_schedular
        self.loss_pointmatching = loss_pointmatching
        self.loss_l1_pos = loss_l1_pos
        self.loss_l1_rot = loss_l1_rot
        self.loss_l1_joints = loss_l1_joints
        self.grasp_repr = grasp_repr
        self.kappa = None if kappa <= 0 else kappa
        self.clip_sample = clip_sample
        self.beta_schedule = beta_schedule
        self.attention = attention
        self.grid_size = grid_size
        self.gripper_name = gripper_name
        self.pose_repr = pose_repr
        self.num_grasps_per_object = num_grasps_per_object
        self.checkpoint_object_encoder_pretrained = checkpoint_object_encoder_pretrained
        self.num_joints = num_joints
        self.num_fingers = num_fingers
        self.condition_on_pose = condition_on_pose

        if self.grasp_repr == "r3_6d":
            self.pose_dim = 9
        elif self.grasp_repr in ["r3_so3", "r3_euler"]:
            self.pose_dim = 6
        else:
            raise NotImplementedError(
                f"Rotation representation {grasp_repr} is not implemented!"
            )
        if self.condition_on_pose:
            # Pose is supplied as a conditioning input (see pose_encoder below),
            # not diffused — the model only ever has to predict joint angles.
            self.output_dim = 2 * self.num_joints
        else:
            # Full TriFinger grasp state: [xyz(3) + rot6d/so3/euler + q_pre(num_joints) + q_final(num_joints)]
            self.output_dim = self.pose_dim + 2 * self.num_joints

        if obs_backbone == "vit":
            from grasp_gen.models.vit import VisionTransformer

            self.object_encoder = VisionTransformer(
                img_size=self.image_size,
                embed_dim=self.num_embed_dim,
                num_classes=self.num_obs_dim,
                patch_size=64,
                depth=12,
                num_heads=8,
            )
        elif obs_backbone == "pointnet":
            self.object_encoder = PointNetPlusPlus(
                output_embedding_dim=self.num_obs_dim,
                feature_dim=1 if self.pose_repr == "pc_feature" else -1,
            )
        elif obs_backbone == "ptv3":
            from grasp_gen.models.ptv3.ptv3 import PointTransformerV3

            self.object_encoder = PointTransformerV3(
                in_channels=3,
                enable_flash=False,
                cls_mode=True,
            )
        else:
            raise NotImplementedError()

        self.diffusion_head = DiffusionNoisePredictionNet(
            diffusion_step_embed_dim=self.diffusion_embed_dim,
            observation_embed_dim=self.num_obs_dim,
            sample_embed_dim=self.diffusion_embed_dim,
            sample_dim=self.output_dim,
            attention=self.attention,
            moreparams=False,
            pose_repr=self.pose_repr,
        )

        if self.condition_on_pose:
            # Projects the (fixed, known) pose's r3_6d encoding into the same
            # width as the object embedding so it can be added as conditioning
            # — cheaper than widening the diffusion head's expected input.
            self.pose_encoder = nn.Sequential(
                nn.Linear(self.pose_dim, self.num_obs_dim),
                nn.ReLU(),
                nn.Linear(self.num_obs_dim, self.num_obs_dim),
            )

        self.contact_heatmap_head = ContactHeatmapHead(
            obs_dim=self.num_obs_dim,
            num_fingers=self.num_fingers,
        )

        if self.condition_on_pose:
            # No pose sub-vector left to split a compositional pos/rot
            # schedule over — always use a single joints-only scheduler.
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule=self.beta_schedule,
                clip_sample=self.clip_sample,
                prediction_type="epsilon",
            )
        elif self.compositional_schedular:
            self.noise_scheduler_pos = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule="scaled_linear",
                clip_sample=True,
                prediction_type="epsilon",
            )

            self.noise_scheduler_rot = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )

        else:
            logger.info(
                f"DDPM parameters, num_diffusion_iters: {self.num_diffusion_iters}, beta_schedule: {self.beta_schedule}, clip_sample: {self.clip_sample}"
            )
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule=self.beta_schedule,  # TODO: Check this
                clip_sample=self.clip_sample,
                prediction_type="epsilon",
            )

        self.gripper_info = get_gripper_info(self.gripper_name)
        self.gripper_mesh = self.gripper_info.collision_mesh
        self.ctr_pts = self.gripper_info.control_points

        # Joint angle normalization: map [lo, hi] → [-1, 1] for the diffusion model.
        # clip_sample=True in DDPMScheduler clips to [-1, 1], so unnormalized radian
        # values (e.g. A_3_joint lo = -2.094) would be clipped without this.
        from grasp_gen.robot import load_default_gripper_config

        _gcfg = load_default_gripper_config(self.gripper_name)
        if "joint_limits" in _gcfg and self.num_joints > 0:
            _limits = np.array(
                _gcfg["joint_limits"], dtype=np.float32
            )  # [num_joints, 2]
            assert _limits.shape == (self.num_joints, 2), (
                f"joint_limits in {self.gripper_name}.yaml has shape {_limits.shape}, "
                f"expected ({self.num_joints}, 2)"
            )
            self.register_buffer("joint_lo", torch.from_numpy(_limits[:, 0]))
            self.register_buffer("joint_hi", torch.from_numpy(_limits[:, 1]))
        else:
            self.joint_lo = None
            self.joint_hi = None

        if self.checkpoint_object_encoder_pretrained is not None:
            if os.path.exists(self.checkpoint_object_encoder_pretrained):
                model_state_dict_object_encoder = load_pretrained_checkpoint_to_dict(
                    self.checkpoint_object_encoder_pretrained, "object_encoder"
                )
                self.object_encoder.load_state_dict(model_state_dict_object_encoder)
                for param in self.object_encoder.parameters():
                    param.requires_grad = False
                logger.info("Using pretrained object encoder!")
            else:
                logger.info(
                    f"Object encoder checkpoints not found at location {self.checkpoint_object_encoder_pretrained}"
                )

    def _normalize_joints(self, q: torch.Tensor) -> torch.Tensor:
        """Map joint angles from radians to [-1, 1] using stored joint limits.

        Args:
            q: [..., num_joints] tensor in radians.
        Returns:
            Same shape tensor in [-1, 1].
        """
        if self.joint_lo is None:
            return q
        lo = self.joint_lo.to(q.device)
        hi = self.joint_hi.to(q.device)
        return 2.0 * (q - lo) / (hi - lo) - 1.0

    def _denormalize_joints(self, q_norm: torch.Tensor) -> torch.Tensor:
        """Map joint angles from [-1, 1] back to radians, clamped to joint limits.

        Args:
            q_norm: [..., num_joints] tensor in [-1, 1].
        Returns:
            Same shape tensor in radians, clamped to [joint_lo, joint_hi].
        """
        if self.joint_lo is None:
            return q_norm
        lo = self.joint_lo.to(q_norm.device)
        hi = self.joint_hi.to(q_norm.device)
        q = (q_norm + 1.0) / 2.0 * (hi - lo) + lo
        return q.clamp(lo, hi)

    @classmethod
    def from_config(cls, cfg):
        """Creates a GraspGenGenerator instance from a configuration object.

        Args:
            cfg: Configuration object containing model parameters

        Returns:
            GraspGenGenerator: Instantiated model
        """
        args = {
            "num_embed_dim": cfg.num_embed_dim,
            "num_obs_dim": cfg.num_obs_dim,
            "diffusion_embed_dim": cfg.diffusion_embed_dim,
            "image_size": cfg.image_size,
            "num_diffusion_iters": cfg.num_diffusion_iters,
            "num_diffusion_iters_eval": cfg.num_diffusion_iters_eval,
            "obs_backbone": cfg.obs_backbone,
            "compositional_schedular": cfg.compositional_schedular,
            "loss_pointmatching": cfg.loss_pointmatching,
            "loss_l1_pos": cfg.loss_l1_pos,
            "loss_l1_rot": cfg.loss_l1_rot,
            "loss_l1_joints": getattr(cfg, "loss_l1_joints", False),
            "grasp_repr": cfg.grasp_repr,
            "kappa": cfg.kappa,
            "clip_sample": cfg.clip_sample,
            "beta_schedule": cfg.beta_schedule,
            "attention": cfg.attention,
            "grid_size": cfg.ptv3.grid_size,
            "gripper_name": cfg.gripper_name,
            "pose_repr": cfg.pose_repr,
            "num_grasps_per_object": cfg.num_grasps_per_object,
            "checkpoint_object_encoder_pretrained": cfg.checkpoint_object_encoder_pretrained,
            "num_joints": getattr(cfg, "num_joints", 8),
            "num_fingers": getattr(cfg, "num_fingers", 3),
            "condition_on_pose": getattr(cfg, "condition_on_pose", False),
        }
        return cls(**args)

    def forward(self, data, cfg=None, eval=False):
        """Forward pass of the model.

        Args:
            data: Input data dictionary containing point clouds and optionally ground truth grasps
            cfg: Optional configuration object
            eval (bool): Whether to run in evaluation mode

        Returns:
            tuple: (outputs, losses, stats) containing model predictions, losses and metrics
        """
        if eval:
            return self.forward_inference(data, return_metrics=True)
        else:
            return self.forward_train(data, cfg=cfg)

    def infer(self, data, return_metrics=False):
        """Inference method for generating grasps.

        Args:
            data: Input data dictionary containing point clouds
            return_metrics (bool): Whether to compute and return evaluation metrics

        Returns:
            tuple: (outputs, losses, stats) containing generated grasps and optional metrics
        """
        return self.forward_inference(data, return_metrics=return_metrics)

    def forward_train(self, data, cfg=None):
        """Training forward pass implementing the diffusion process.

        Args:
            data: Input data dictionary containing point clouds and ground truth grasps

        Returns:
            tuple: (outputs, losses, stats) containing predictions, training losses and metrics
        """
        device = data["points"].device
        num_objects_in_batch = len(data["points"])

        depth = data["points"]
        grasps = data["grasps"]

        num_points = depth.shape[-2]
        depth = depth.reshape([-1, num_points, 3])

        if isinstance(grasps, list):
            grasps = torch.stack(grasps)  # [B, N, 4, 4] — cat gave [B*N, 4, 4] and shape[1]=4 (wrong)

        # grasps is now [num_objects_in_batch, num_grasps_per_object, 4, 4]
        num_grasps_per_batch = grasps.shape[1]
        batch_size = num_objects_in_batch * num_grasps_per_batch
        grasps_init_size = [num_objects_in_batch, num_grasps_per_batch, 4, 4]

        grasps = grasps.reshape([-1, 4, 4])

        if self.kappa is not None:
            depth = self.kappa * depth

        # Save xyz before potential ptv3 restructuring — needed by heatmap head
        depth_xyz = depth  # [num_objects_in_batch, N, 3]

        if self.obs_backbone == "ptv3":
            depth = convert_to_ptv3_pc_format(depth, grid_size=self.grid_size)

        # Build 25D TriFinger grasp state: [pose_9d | q_pre | q_final]
        pose_9d = matrix_to_rt(grasps, self.grasp_repr, kappa=self.kappa)

        if "q_pre" in data:
            q_pre = data["q_pre"]
            q_final = data["q_final"]
            if isinstance(q_pre, list):
                q_pre = torch.cat(q_pre)
            if isinstance(q_final, list):
                q_final = torch.cat(q_final)
            q_pre = q_pre.reshape([-1, self.num_joints]).to(device)
            q_final = q_final.reshape([-1, self.num_joints]).to(device)
        else:
            q_pre = torch.zeros([batch_size, self.num_joints], device=device)
            q_final = torch.zeros([batch_size, self.num_joints], device=device)

        # Normalize joints from radians to [-1, 1] so DDPMScheduler clip_sample
        # does not truncate values that legitimately exceed ±1 in radian space.
        q_pre = self._normalize_joints(q_pre)
        q_final = self._normalize_joints(q_final)

        if self.condition_on_pose:
            # Pose is conditioning, not a diffusion target — only joints are diffused.
            grasps_gt = torch.cat([q_pre, q_final], dim=-1)
        else:
            grasps_gt = components_to_grasp_state(pose_9d, q_pre, q_final)

        noise = torch.randn([batch_size, self.output_dim], device=device).float()

        timesteps = torch.randint(
            0, self.num_diffusion_iters, (batch_size,), device=device
        ).long()

        offset = (
            torch.tensor([num_grasps_per_batch])
            .repeat(num_objects_in_batch)
            .cumsum(dim=0)
            .to(device)
        )
        mask_batch = offset2batch(offset)

        if self.pose_repr in ["grasp_cloud", "grasp_cloud_pe", "pc_feature"]:
            ctrl_pts = self.ctr_pts.to(device=device)
            grasp_pc = (grasps @ ctrl_pts).transpose(-2, -1)[..., :3]

        if self.pose_repr == "pc_feature":
            depth_full = depth[mask_batch]
            depth_full = torch.cat([depth_full, grasp_pc], dim=1)
            pc_feature = torch.cat(
                [
                    torch.zeros(
                        [num_grasps_per_batch * num_objects_in_batch, num_points, 1]
                    ),
                    torch.ones(
                        [
                            num_grasps_per_batch * num_objects_in_batch,
                            grasp_pc.shape[1],
                            1,
                        ]
                    ),
                ],
                dim=1,
            ).to(device=device)

            object_embedding = torch.cat([depth_full, pc_feature], dim=-1)
            object_embedding = self.object_encoder(object_embedding)
        elif self.pose_repr == "mlp":
            object_embedding = self.object_encoder(
                depth
            )  # object_embedding size is [num_objects_in_batch, self.num_obs_dim]
            per_obj_embedding = (
                object_embedding  # save before redistribution for heatmap head
            )
            object_embedding = object_embedding[
                mask_batch
            ]  # Redistribute object embeddings to full batch, result is [batch_size, self.num_obs_dim]
        else:
            per_obj_embedding = None
            raise NotImplementedError(f"Pose repr {self.pose_repr} not implemented!")

        if self.condition_on_pose:
            # Fuse the (fixed, known) target pose into the conditioning —
            # additive fusion keeps the diffusion head's expected input width
            # unchanged.
            object_embedding = object_embedding + self.pose_encoder(pose_9d)

        if self.condition_on_pose:
            noisy_grasps = self.noise_scheduler.add_noise(grasps_gt, noise, timesteps)
        elif self.compositional_schedular:
            noisy_grasps_pos = self.noise_scheduler_pos.add_noise(
                grasps_gt[..., :3], noise[..., :3], timesteps
            )
            # rot scheduler covers rotation + joint dims (everything after xyz)
            noisy_grasps_rot = self.noise_scheduler_rot.add_noise(
                grasps_gt[..., 3:],
                noise[..., 3:],
                timesteps,
            )
            noisy_grasps = torch.hstack([noisy_grasps_pos, noisy_grasps_rot])
        else:
            noisy_grasps = self.noise_scheduler.add_noise(grasps_gt, noise, timesteps)

        samples = noisy_grasps if self.pose_repr == "mlp" else None
        noise_pred = self.diffusion_head(object_embedding, timesteps, samples)

        if self.condition_on_pose:
            # No pose sub-vector in noise_pred/noisy_grasps/grasps_gt at all —
            # the entire output is joint-angle noise. Pose-accuracy stats and
            # the pose point-matching loss are moot (pose is a fixed input,
            # not a prediction), so skip them; joint L1 is the sole signal.
            stats = {}
            losses = {}
            joint_loss = torch.linalg.norm(noise - noise_pred, dim=-1)
            losses["joint_loss"] = (1.0, torch.mean(joint_loss))
        else:
            pred_noise_pts_mat = rt_to_matrix(
                noise_pred[:, : self.pose_dim], self.grasp_repr, self.kappa
            )
            actual_noise_pts_mat = rt_to_matrix(
                noise[:, : self.pose_dim], self.grasp_repr, self.kappa
            )
            noisy_grasps_mat = rt_to_matrix(
                noisy_grasps[:, : self.pose_dim], self.grasp_repr, self.kappa
            )
            grasps_gt_mat = rt_to_matrix(
                grasps_gt[:, : self.pose_dim], self.grasp_repr, self.kappa
            )

            stats = compute_metrics_given_two_sets_of_poses(
                actual_noise_pts_mat, pred_noise_pts_mat, self.gripper_info
            )

            losses = {}
            if self.loss_pointmatching:
                point_matching_loss = compute_grasp_loss(
                    actual_noise_pts_mat, pred_noise_pts_mat, self.ctr_pts
                )
                losses["noise_pred"] = (2.0, point_matching_loss)

            if self.loss_l1_pos:
                position_loss = torch.linalg.norm(
                    noise[..., :3] - noise_pred[..., :3], dim=-1
                )
                position_loss = torch.mean(position_loss)  # across the batch
                losses["position_loss"] = (1.0, position_loss)

            if self.loss_l1_rot:
                rotation_loss = torch.linalg.norm(
                    noise[..., 3 : self.pose_dim] - noise_pred[..., 3 : self.pose_dim],
                    dim=-1,
                )
                rotation_loss = torch.mean(rotation_loss)
                losses["rotation_loss"] = (1.0, rotation_loss)

                # Joint losses (q_pre and q_final in the diffusion noise space)
                joint_loss = torch.linalg.norm(
                    noise[..., self.pose_dim :] - noise_pred[..., self.pose_dim :],
                    dim=-1,
                )
                joint_loss = torch.mean(joint_loss)
                losses["joint_loss"] = (1.0, joint_loss)

            if self.loss_l1_joints and self.pose_dim < self.output_dim:
                joint_loss = torch.linalg.norm(
                    noise[..., self.pose_dim :] - noise_pred[..., self.pose_dim :],
                    dim=-1,
                )
                losses["joint_loss"] = (1.0, torch.mean(joint_loss))

        grasp_state_init_size = [
            num_objects_in_batch,
            num_grasps_per_batch,
            self.output_dim,
        ]
        outputs = {}
        if not self.condition_on_pose:
            outputs["actual_noise_pts_mat"] = actual_noise_pts_mat.reshape(grasps_init_size)
            outputs["pred_noise_pts_mat"] = pred_noise_pts_mat.reshape(grasps_init_size)

            outputs["noisy_grasps_mat"] = noisy_grasps_mat.reshape(grasps_init_size)
            outputs["grasps_gt_mat"] = grasps_gt_mat.reshape(grasps_init_size)

        # Full 25D state outputs
        outputs["grasp_state_gt"] = grasps_gt.reshape(grasp_state_init_size)
        outputs["grasp_state_noisy"] = noisy_grasps.reshape(grasp_state_init_size)

        # Contact heatmap prediction (only for mlp pose_repr where per_obj_embedding is available)
        if per_obj_embedding is not None:
            heatmap_pred = self.contact_heatmap_head(depth_xyz, per_obj_embedding)
            outputs["contact_heatmap_pred"] = heatmap_pred.sigmoid()
            use_heatmap = getattr(cfg, "use_contact_heatmap", False)
            if use_heatmap and "contact_heatmap" in data:
                heatmap_gt = data["contact_heatmap"]
                if isinstance(heatmap_gt, list):
                    heatmap_gt = torch.stack(heatmap_gt)
                heatmap_gt = heatmap_gt.to(device).float()
                lambda_h = getattr(cfg, "lambda_contact_heatmap", 1.0)
                if "has_contact_heatmap" in data:
                    # Per-sample mask: only supervise samples with real contact data
                    mask = data["has_contact_heatmap"].to(device)
                    if mask.any():
                        losses["contact_heatmap"] = (
                            lambda_h,
                            focal_loss_with_logits(
                                heatmap_pred[mask], heatmap_gt[mask]
                            ),
                        )
                elif heatmap_gt.sum() > 0:
                    # Backward compat: no has_contact_heatmap field — use old sum>0 guard
                    losses["contact_heatmap"] = (
                        lambda_h,
                        focal_loss_with_logits(heatmap_pred, heatmap_gt),
                    )

        return outputs, losses, stats

    def forward_inference(self, data, return_metrics=False):
        """Inference forward pass implementing the reverse diffusion process.

        Args:
            data: Input data dictionary containing point clouds
            return_metrics (bool): Whether to compute evaluation metrics

        Returns:
            tuple: (outputs, losses, stats) containing generated grasps and optional metrics
        """

        device = data["points"].device

        num_objects_in_batch = len(data["points"])
        if "grasps" in data:
            if isinstance(data["grasps"][0], list):
                data["grasps"][0] = np.array(data["grasps"][0])
            num_grasps_per_batch = data["grasps"][0].shape[0]
        else:
            num_grasps_per_batch = self.num_grasps_per_object
            return_metrics = False

        if self.condition_on_pose:
            # Pose is supplied, not generated — data["grasps"] carries the
            # TARGET poses to condition on (reused directly as T_palm below),
            # not a diffusion target to denoise toward.
            assert "grasps" in data, (
                "condition_on_pose=True requires data['grasps'] "
                "(the target poses to hold fixed during generation)."
            )
            target_grasps = data["grasps"]
            if isinstance(target_grasps, list):
                target_grasps = torch.stack(target_grasps)
            target_grasps = target_grasps.reshape([-1, 4, 4]).to(device).float()
            target_pose_9d = matrix_to_rt(target_grasps, self.grasp_repr, kappa=self.kappa)

        batch_size = data["points"].shape[0] * num_grasps_per_batch
        depth = data["points"]

        num_points = depth.shape[-2]

        depth = depth.reshape([-1, num_points, 3])
        depth = depth.to(device)

        grasps_init_size = [num_objects_in_batch, num_grasps_per_batch, 4, 4]
        grasp_state_init_size = [
            num_objects_in_batch,
            num_grasps_per_batch,
            self.output_dim,
        ]

        if self.kappa is not None:
            depth = self.kappa * depth

        depth_xyz = depth  # save xyz before potential ptv3 restructuring

        if self.obs_backbone == "ptv3":
            depth = convert_to_ptv3_pc_format(depth, grid_size=self.grid_size)

        # Store per-iteration pose (4x4) for visualisation; joints stored separately
        grasps_per_iteration = torch.zeros(
            [
                num_objects_in_batch,
                self.num_diffusion_iters_eval,
                num_grasps_per_batch,
                4,
                4,
            ]
        )

        with torch.no_grad():
            noisy_init = torch.randn([batch_size, self.output_dim], device=device)
            noisy_grasps = noisy_init

            # Initialize likelihood scores
            likelihood = torch.zeros((batch_size, 1), device=device)

            offset = (
                torch.tensor([num_grasps_per_batch])
                .repeat(num_objects_in_batch)
                .cumsum(dim=0)
                .to(device)
            )
            mask_batch = offset2batch(offset)

            if self.pose_repr == "mlp":
                object_embedding = self.object_encoder(
                    depth
                )  # object_embedding size is [num_objects_in_batch, self.num_obs_dim]
                per_obj_embedding = (
                    object_embedding  # save before redistribution for heatmap head
                )
                object_embedding = object_embedding[
                    mask_batch
                ]  # Redistribute object embeddings to full batch, result is [batch_size, self.num_obs_dim]

            if self.condition_on_pose:
                object_embedding = object_embedding + self.pose_encoder(target_pose_9d)

            if self.condition_on_pose:
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters_eval)
                timesteps = self.noise_scheduler.timesteps
            elif self.compositional_schedular:
                self.noise_scheduler_pos.set_timesteps(self.num_diffusion_iters_eval)
                timesteps = self.noise_scheduler_pos.timesteps
                self.noise_scheduler_rot.set_timesteps(self.num_diffusion_iters_eval)
            else:
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters_eval)
                timesteps = self.noise_scheduler.timesteps

            for iter_idx, k in enumerate(timesteps):
                samples = noisy_grasps if self.pose_repr == "mlp" else None

                if self.pose_repr in ["grasp_cloud", "grasp_cloud_pe", "pc_feature"]:
                    ctrl_pts = self.ctr_pts.to(device=device)

                    noisy_grasps_mat = rt_to_matrix(
                        noisy_grasps[:, : self.pose_dim], self.grasp_repr, self.kappa
                    )
                    grasp_pc = (noisy_grasps_mat @ ctrl_pts).transpose(-2, -1)[..., :3]

                if self.pose_repr == "pc_feature":
                    depth_full = depth[mask_batch]
                    depth_full = torch.cat([depth_full, grasp_pc], dim=1)
                    pc_feature = torch.cat(
                        [
                            torch.zeros(
                                [
                                    num_grasps_per_batch * num_objects_in_batch,
                                    num_points,
                                    1,
                                ]
                            ),
                            torch.ones(
                                [
                                    num_grasps_per_batch * num_objects_in_batch,
                                    grasp_pc.shape[1],
                                    1,
                                ]
                            ),
                        ],
                        dim=1,
                    ).to(device=device)

                    object_embedding = torch.cat([depth_full, pc_feature], dim=-1)
                    object_embedding = self.object_encoder(object_embedding)

                # Forward: Predict noise
                noise_pred = self.diffusion_head(object_embedding, k, samples)

                if not self.condition_on_pose and self.compositional_schedular:
                    # pos scheduler: xyz dims; rot scheduler: rotation + joint dims
                    res_pos = self.noise_scheduler_pos.step(
                        model_output=noise_pred[..., :3],
                        timestep=k,
                        sample=noisy_grasps[..., :3],
                    )
                    res_rot = self.noise_scheduler_rot.step(
                        model_output=noise_pred[..., 3:],
                        timestep=k,
                        sample=noisy_grasps[..., 3:],
                    )

                    # Compute likelihood contributions
                    if k > 0:  # Skip first step
                        beta_pos = self.noise_scheduler_pos.betas[k]
                        likelihood_pos = (
                            torch.distributions.Normal(
                                res_pos.pred_original_sample,
                                torch.sqrt(torch.tensor(beta_pos, device=device)),
                            )
                            .log_prob(noisy_grasps[..., :3])
                            .sum(-1, keepdim=True)
                        )

                        beta_rot = self.noise_scheduler_rot.betas[k]
                        likelihood_rot = (
                            torch.distributions.Normal(
                                res_rot.pred_original_sample,
                                torch.sqrt(torch.tensor(beta_rot, device=device)),
                            )
                            .log_prob(noisy_grasps[..., 3:])
                            .sum(-1, keepdim=True)
                        )

                        likelihood += likelihood_pos + likelihood_rot

                    noisy_grasps = torch.hstack(
                        [res_pos.prev_sample, res_rot.prev_sample]
                    )
                else:
                    # Handle standard case
                    res = self.noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=noisy_grasps
                    )

                    # Compute likelihood contribution
                    if k > 0:  # Skip first step
                        beta = self.noise_scheduler.betas[k]
                        var = beta
                        likelihood += (
                            torch.distributions.Normal(
                                res.pred_original_sample,
                                torch.sqrt(torch.tensor(var, device=device)),
                            )
                            .log_prob(noisy_grasps)
                            .sum(-1, keepdim=True)
                        )

                    noisy_grasps = res.prev_sample

                if self.condition_on_pose:
                    grasps_pred = target_grasps.reshape(grasps_init_size)
                else:
                    pred_grasps = rt_to_matrix(
                        noisy_grasps[:, : self.pose_dim], self.grasp_repr, self.kappa
                    )
                    grasps_pred = pred_grasps.reshape(grasps_init_size)

                grasps_per_iteration[:, iter_idx, :, ::] = grasps_pred

        # Final pose matrix
        if self.condition_on_pose:
            # Pose was held fixed throughout — it IS the supplied target, not
            # something recovered from noisy_grasps (which holds joints only).
            T_palm = target_grasps.reshape(grasps_init_size)
        else:
            T_palm = rt_to_matrix(
                noisy_grasps[:, : self.pose_dim], self.grasp_repr, self.kappa
            )
            T_palm = T_palm.reshape(grasps_init_size)
            T_palm[:, :, 3, 3] = 1  # proper homogeneous matrix

        # Split joint dims from the final denoised state and convert back to radians.
        # noisy_grasps contains normalized joints ([-1, 1]); denormalize to physical units.
        joint_pose_dim = 0 if self.condition_on_pose else self.pose_dim
        q_pre_out = noisy_grasps[:, joint_pose_dim : joint_pose_dim + self.num_joints]
        q_final_out = noisy_grasps[:, joint_pose_dim + self.num_joints : self.output_dim]
        q_pre_out = self._denormalize_joints(q_pre_out)
        q_final_out = self._denormalize_joints(q_final_out)
        q_pre_out = q_pre_out.reshape(
            [num_objects_in_batch, num_grasps_per_batch, self.num_joints]
        )
        q_final_out = q_final_out.reshape(
            [num_objects_in_batch, num_grasps_per_batch, self.num_joints]
        )

        grasp_state_out = noisy_grasps.reshape(grasp_state_init_size)

        grasps_pred = T_palm

        stats_batch = []

        if return_metrics:
            all_stats = []
            for i in range(num_objects_in_batch):
                grasps_pred_i = grasps_pred[i].cpu().numpy()
                grasps_gt_i = data["grasps_highres"][i].cpu().numpy()

                tree = KDTree(grasps_gt_i[:, :3, 3])
                dist, idx = tree.query(grasps_pred_i[:, :3, 3])
                matched = dist < 4.0
                idx = idx[matched]

                grasps_pred_matched = grasps_pred_i[matched]
                grasps_gt_for_pred = grasps_gt_i[idx]

                grasps_pred_matched = torch.from_numpy(grasps_pred_matched)
                grasps_gt_for_pred = torch.from_numpy(grasps_gt_for_pred)
                stats = compute_metrics_given_two_sets_of_poses(
                    grasps_gt_for_pred,
                    grasps_pred_matched,
                    self.gripper_info,
                    consider_symmetry=True,
                )

                recall = compute_recall(grasps_gt_i, grasps_pred_i)
                precision = compute_recall(grasps_pred_i, grasps_gt_i)

                stats["recall"] = torch.tensor(recall).to(device)
                stats["precision"] = torch.tensor(precision).to(device)

                all_stats.append(stats)

            stats_keys = all_stats[0].keys()
            stats_batch = {}

            for key in stats_keys:
                stats_batch[key] = torch.mean(
                    torch.tensor([stats[key] for stats in all_stats]).to(device)
                )

        outputs = {
            # TriFinger action state components
            "T_palm": T_palm,
            "q_pre": q_pre_out,
            "q_final": q_final_out,
            "grasp_state": grasp_state_out,
            # Legacy key kept for downstream inference scripts that expect grasps_pred as 4x4
            "grasps_pred": grasps_pred,
            "grasps_per_iteration": grasps_per_iteration,
            "grasp_confidence": torch.zeros(grasps_pred.shape[0]),
            "grasping_masks": torch.zeros(grasps_pred.shape[0]),
            "grasp_contacts": torch.zeros(grasps_pred.shape[0]),
            "instance_masks": torch.zeros(grasps_pred.shape[0]),
            "likelihood": likelihood.reshape(
                num_objects_in_batch, num_grasps_per_batch, 1
            ),
        }

        # Contact heatmap — exposed at inference for debugging / downstream use
        if self.pose_repr == "mlp":
            heatmap = self.contact_heatmap_head(depth_xyz, per_obj_embedding)
            outputs["contact_heatmap"] = (
                heatmap.sigmoid()
            )  # [num_objects, N, num_fingers]

        return outputs, {}, stats_batch


class DiffusionNoisePredictionNet(nn.Module):
    """Neural network module implementing the diffusion model's denoising network.

    This network predicts the noise in the diffused samples given the current noisy sample,
    diffusion timestep, and object observation embedding.

    Args:
        diffusion_step_embed_dim (int): Dimension for diffusion step embeddings. Default: 512
        observation_embed_dim (int): Dimension of object observation embeddings. Default: 512
        sample_embed_dim (int): Dimension for sample embeddings. Default: 512
        sample_dim (int): Dimension of the grasp samples. Default: 9
        moreparams (bool): Whether to use additional parameters. Default: False
        attention (bool): Whether to use attention mechanisms. Default: False
        pose_repr (str): Type of pose representation. Default: 'mlp'
    """

    def __init__(
        self,
        diffusion_step_embed_dim=512,
        observation_embed_dim=512,
        sample_embed_dim=512,
        sample_dim=9,
        moreparams=False,
        attention=False,
        pose_repr="mlp",
    ):

        self.attention = attention
        self.pose_repr = pose_repr
        super().__init__()

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        if self.pose_repr == "mlp":
            self.sample_encoder = nn.Sequential(
                nn.Linear(sample_dim, sample_embed_dim),
                nn.ReLU(),
                nn.Linear(sample_embed_dim, sample_embed_dim),
            )

            total_input_dim = (
                sample_embed_dim + diffusion_step_embed_dim + observation_embed_dim
            )
        else:
            total_input_dim = diffusion_step_embed_dim + observation_embed_dim

        self.prediction_head = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim // 2),
            nn.ReLU(),
            nn.Linear(total_input_dim // 2, total_input_dim // 4),
            nn.ReLU(),
            nn.Linear(total_input_dim // 4, sample_dim),
        )

        if self.attention.find("attn") > 0:
            from grasp_gen.models.model_utils import AttentionLayer, FFNLayer

            # transformer decoder
            self.embed_dim = total_input_dim
            self.num_heads = 8
            self.num_layers = 3
            self.feedforward_dim = 512
            self.activation = "GELU"
            num_grasp_queries = 1

            if self.attention.find("cross") >= 0:
                self.obs_pos_enc = nn.Embedding(1, observation_embed_dim)
                self.sample_pos_enc = nn.Embedding(1, sample_embed_dim)

                self.time_pos_enc = nn.Embedding(1, diffusion_step_embed_dim)

                self.query_embed = nn.Embedding(1, self.embed_dim)
                self.query_pos_enc = nn.Embedding(1, self.embed_dim)
                self.self_attention_layers = nn.ModuleList()
                self.cross_attention_layers = nn.ModuleList()
                self.ffn_layers = nn.ModuleList()
                for _ in range(self.num_layers):
                    self.self_attention_layers.append(
                        AttentionLayer(self.embed_dim, self.num_heads)
                    )
                    self.cross_attention_layers.append(
                        AttentionLayer(self.embed_dim, self.num_heads)
                    )
                    self.ffn_layers.append(
                        FFNLayer(self.embed_dim, self.feedforward_dim, self.activation)
                    )

            else:
                self.query_pos_enc = nn.Embedding(num_grasp_queries, self.embed_dim)

                self.self_attention_layers = nn.ModuleList()
                self.ffn_layers = nn.ModuleList()
                for _ in range(self.num_layers):
                    self.self_attention_layers.append(
                        AttentionLayer(self.embed_dim, self.num_heads)
                    )
                    self.ffn_layers.append(
                        FFNLayer(self.embed_dim, self.feedforward_dim, self.activation)
                    )

    def forward(
        self,
        observation_embedding: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor = None,
    ):
        """Forward pass of the diffusion denoising network.

        Args:
            observation_embedding (torch.Tensor): Object observation embeddings
            timesteps (torch.Tensor): Current diffusion timesteps
            sample (torch.Tensor, optional): Current noisy samples

        Returns:
            torch.Tensor: Predicted noise in the samples
        """

        device = observation_embedding.device

        if torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
            timesteps = timesteps.expand(observation_embedding.shape[0])

        timestep_embedding = self.diffusion_step_encoder(timesteps)

        if self.pose_repr == "mlp":
            sample_embedding = self.sample_encoder(sample)

        if self.attention.find("attn") >= 0:
            if self.attention.find("cross") >= 0:
                from grasp_gen.models.model_utils import repeat_new_axis

                # t0 = time.time()

                batch_size = timestep_embedding.shape[0]

                embed = repeat_new_axis(self.query_embed.weight, batch_size, dim=1)
                query_pos_enc = repeat_new_axis(
                    self.query_pos_enc.weight, batch_size, dim=1
                )

                obs_pos_enc = repeat_new_axis(
                    self.obs_pos_enc.weight, batch_size, dim=1
                )

                time_pos_enc = repeat_new_axis(
                    self.time_pos_enc.weight, batch_size, dim=1
                )

                # TODO - Improve this...
                sample_pos_enc = repeat_new_axis(
                    self.sample_pos_enc.weight, batch_size, dim=1
                )

                cross_embed = torch.cat(
                    [timestep_embedding, observation_embedding, sample_embedding],
                    axis=1,
                ).unsqueeze(0)
                cross_embed_pos_enc = torch.cat(
                    [time_pos_enc, obs_pos_enc, sample_pos_enc], axis=2
                )

                for i in range(self.num_layers):
                    embed = self.cross_attention_layers[i](
                        embed,
                        cross_embed,
                        cross_embed + cross_embed_pos_enc,
                        query_pos_enc,
                        cross_embed_pos_enc,
                    )

                    embed = self.self_attention_layers[i](
                        embed,
                        embed,
                        embed + query_pos_enc,
                        query_pos_enc,
                        query_pos_enc,
                    )
                    embed = self.ffn_layers[i](embed)
            else:
                from grasp_gen.models.model_utils import repeat_new_axis

                if self.pose_repr == "mlp":
                    embed = torch.cat(
                        [sample_embedding, timestep_embedding, observation_embedding],
                        axis=-1,
                    )
                    # print(f"Concatenation took {time.time() - t0}s")
                else:
                    embed = torch.cat(
                        [timestep_embedding, observation_embedding], axis=-1
                    )
                embed = embed.unsqueeze(0)
                batch_size = embed.shape[1]

                query_pos_enc = repeat_new_axis(
                    self.query_pos_enc.weight, batch_size, dim=1
                )

                for i in range(self.num_layers):
                    embed = self.self_attention_layers[i](
                        embed,
                        embed,
                        embed + query_pos_enc,
                        query_pos_enc,
                        query_pos_enc,
                    )
                    embed = self.ffn_layers[i](embed)
                # print(f"Attention took {time.time() - t0}s")
            embed = embed.squeeze(0)
        else:
            if self.pose_repr == "mlp":
                embed = torch.cat(
                    [sample_embedding, timestep_embedding, observation_embedding],
                    axis=-1,
                )
            else:
                embed = torch.cat([timestep_embedding, observation_embedding], axis=-1)

        return self.prediction_head(embed)
