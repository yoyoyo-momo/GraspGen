import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
from pathlib import Path

from grasp_gen.robot import load_control_points_core, load_default_gripper_config


class GripperModel:
    """Procedural TriFinger mesh — palm disc + three finger cylinders + fingertip spheres."""

    def __init__(self, data_root_dir=None):
        cfg = load_default_gripper_config("trifinger")
        cps = np.array(cfg["control_points"], dtype=np.float64)  # [3, 3]

        palm = trimesh.creation.cylinder(radius=0.04, height=0.01, sections=32)

        parts = [palm]
        for tip in cps:
            length = float(np.linalg.norm(tip))
            center = tip / 2.0
            direction = tip / length

            cyl = trimesh.creation.cylinder(radius=0.008, height=length, sections=8)
            # Rotate cylinder from default z-axis orientation to finger direction
            z = np.array([0.0, 0.0, 1.0])
            cross = np.cross(z, direction)
            if np.linalg.norm(cross) > 1e-6:
                rot_axis = cross / np.linalg.norm(cross)
                angle = float(np.arccos(np.clip(np.dot(z, direction), -1.0, 1.0)))
                R = tra.rotation_matrix(angle, rot_axis)
            else:
                R = np.eye(4)
            R[:3, 3] = center
            cyl.apply_transform(R)
            parts.append(cyl)

            tip_sphere = trimesh.creation.icosphere(radius=0.012)
            tip_sphere.apply_translation(tip)
            parts.append(tip_sphere)

        self.mesh = trimesh.util.concatenate(parts)

    def get_gripper_collision_mesh(self):
        return self.mesh

    def get_gripper_visual_mesh(self):
        return self.mesh


def load_control_points() -> torch.Tensor:
    """3 fingertip + origin control points as a homogeneous [4, 4] tensor."""
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = np.array(
        load_control_points_core(gripper_config), dtype=np.float32
    )  # [3, 3]
    control_points = np.vstack([control_points, np.zeros(3)])  # [4, 3]
    control_points = np.hstack([control_points, np.ones((4, 1))])  # [4, 4]
    return torch.from_numpy(control_points).float().T  # [4, 4]


def load_control_points_for_visualization():
    """Three line segments from palm origin to each fingertip."""
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    cps = np.array(load_control_points_core(gripper_config), dtype=np.float32)  # [3, 3]
    origin = [0.0, 0.0, 0.0]
    return [
        [origin, cps[0].tolist()],  # A
        [origin, cps[1].tolist()],  # B
        [origin, cps[2].tolist()],  # C
    ]
