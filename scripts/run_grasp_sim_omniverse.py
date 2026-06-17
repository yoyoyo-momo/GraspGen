# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Isaac Sim script: after opening box_with_grasps_sim.usd, run this so that on
Play the grippers close, grasp the object, and per-finger contact positions are
recorded and written to a JSON sidecar file.

Usage in Isaac Sim:
  1. Open the stage: File > Open > box_with_grasps_sim.usd
  2. Open Script Editor (Window > Script Editor)
  3. Run this script (e.g. paste and Run, or run as standalone with isaacsim python)
  4. Press Play – grippers will move to closed position after a short delay,
     then contacts are sampled and written to CONTACT_OUTPUT_PATH.

Requires: Isaac Sim with omni.isaac.core and omni.physx (bundled).
"""

import json
import os

import numpy as np

# Isaac Sim / Omniverse imports (available when run inside Isaac Sim)
try:
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.types import ArticulationAction

    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Delay after Play (s) before commanding grippers to close
GRASP_CLOSE_DELAY_S = 0.5

# Additional wait after close before sampling contacts (let simulation settle)
CONTACT_SETTLE_DELAY_S = 1.0

# TriFinger fingertip link names (validated against the TriFinger USD asset)
TRIFINGER_FINGER_LINKS = ["A_3_Link", "B_3_Link", "C_3_Link"]

# Where to write the contact record JSON.
# Override this before running if your workspace differs.
CONTACT_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "contact_record.json"
)

# For Robotiq-style grippers: use lower joint limit as "closed" (fingers close when going to min).
CLOSED_JOINT_LIMIT = "lower"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_articulation_roots_under_world(stage):
    """Return prim paths of articulation roots under /World (Env_*/gripper)."""
    roots = []
    world = stage.GetPrimAtPath("/World")
    if not world:
        return roots
    for env in world.GetChildren():
        if not env.GetName().startswith("Env_"):
            continue
        gripper_path = env.GetPath().pathString + "/gripper"
        gripper = stage.GetPrimAtPath(gripper_path)
        if gripper and gripper.IsValid():
            roots.append(gripper_path)
    return roots


def _get_closed_joint_positions(articulation, use_lower=True):
    """Get joint positions for 'closed' grasp from articulation limits."""
    joint_names = articulation.dof_names
    limits = articulation.get_joint_limits()
    if limits is None or len(limits) != len(joint_names):
        return None
    pos = []
    for low, high in limits:
        pos.append(float(low) if use_lower else float(high))
    return np.array(pos, dtype=np.float64)


def _extract_env_index(prim_path: str):
    """Parse the Env_N index out of a USD prim path, or return None."""
    for part in prim_path.split("/"):
        if part.startswith("Env_"):
            try:
                return int(part[4:])
            except ValueError:
                pass
    return None


def _build_contact_recorder(num_envs: int, finger_links: list):
    """Return (callback_fn, env_contacts) for per-finger contact accumulation.

    env_contacts[env_idx][finger_idx] = [[x, y, z], ...]
    The callback must be registered with world.add_timestep_callback().
    """
    try:
        from omni.physx import get_physx_interface

        physx = get_physx_interface()
    except ImportError:
        print(
            "run_grasp_sim_omniverse: omni.physx not available — contact recording disabled."
        )
        return None, None

    env_contacts = {
        i: {f: [] for f in range(len(finger_links))} for i in range(num_envs)
    }

    def _record_contacts(dt):
        headers, data = physx.get_contact_report()
        data_offset = 0
        for header in headers:
            a0 = str(header.actor0)
            a1 = str(header.actor1)

            for f_idx, link_name in enumerate(finger_links):
                finger_in_a0 = link_name in a0
                finger_in_a1 = link_name in a1
                if not (finger_in_a0 or finger_in_a1):
                    continue

                finger_path = a0 if finger_in_a0 else a1
                env_idx = _extract_env_index(finger_path)
                if env_idx is None or env_idx not in env_contacts:
                    continue

                for k in range(header.num_contact_data):
                    cp = data[data_offset + k].position
                    env_contacts[env_idx][f_idx].append(
                        [float(cp[0]), float(cp[1]), float(cp[2])]
                    )

            data_offset += header.num_contact_data

    return _record_contacts, env_contacts


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_grasp_on_play():
    """Register timestep callbacks so that after Play:
    1. Grippers close (at GRASP_CLOSE_DELAY_S).
    2. Contacts are sampled (at GRASP_CLOSE_DELAY_S + CONTACT_SETTLE_DELAY_S).
    3. Contact JSON is written to CONTACT_OUTPUT_PATH.
    """
    if not ISAAC_AVAILABLE:
        print(
            "run_grasp_sim_omniverse: omni.isaac.core not available. Run inside Isaac Sim."
        )
        return

    from omni.isaac.core.utils.stage import get_current_stage
    import carb

    stage = get_current_stage()
    if not stage:
        print(
            "run_grasp_sim_omniverse: No stage open. Open box_with_grasps_sim.usd first."
        )
        return

    root_paths = _get_articulation_roots_under_world(stage)
    if not root_paths:
        print(
            "run_grasp_sim_omniverse: No /World/Env_*/gripper articulation roots found."
        )
        return

    world = World.instance()
    if world is None:
        print(
            "run_grasp_sim_omniverse: World not initialized. Press Play once, then re-run."
        )
        return

    articulations = []
    for path in root_paths:
        art = world.scene.get(path)
        if art is None:
            art = Articulation(path)
            world.scene.add(art)
        articulations.append(art)

    num_envs = len(root_paths)
    contact_callback, env_contacts = _build_contact_recorder(
        num_envs, TRIFINGER_FINGER_LINKS
    )

    # ------------------------------------------------------------------
    # Phase 1: close grippers
    # ------------------------------------------------------------------
    def _close_grippers():
        for art in articulations:
            if art is None:
                continue
            try:
                closed_pos = _get_closed_joint_positions(
                    art, use_lower=(CLOSED_JOINT_LIMIT == "lower")
                )
                if closed_pos is not None:
                    art.apply_action(ArticulationAction(joint_positions=closed_pos))
            except Exception as e:
                carb.log_warn(f"run_grasp_sim_omniverse close_grippers: {e}")

    # ------------------------------------------------------------------
    # Phase 2: sample contacts and save
    # ------------------------------------------------------------------
    def _save_contacts():
        if env_contacts is None:
            return
        world.remove_timestep_callback("record_contacts")

        output = {
            str(env_idx): {
                str(f_idx): contacts for f_idx, contacts in finger_data.items()
            }
            for env_idx, finger_data in env_contacts.items()
        }
        output_path = os.path.abspath(CONTACT_OUTPUT_PATH)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(output, fh, indent=2)

        total = sum(len(c) for fd in env_contacts.values() for c in fd.values())
        print(
            f"run_grasp_sim_omniverse: Saved {total} contact points across "
            f"{num_envs} env(s) to {output_path}"
        )

    # ------------------------------------------------------------------
    # Unified timestep callback (two-phase timer)
    # ------------------------------------------------------------------
    elapsed = [0.0]
    phase = [0]  # 0 = waiting to close, 1 = waiting to sample

    def _on_timestep(dt):
        elapsed[0] += dt

        if phase[0] == 0 and elapsed[0] >= GRASP_CLOSE_DELAY_S:
            phase[0] = 1
            elapsed[0] = 0.0
            _close_grippers()
            # Start contact recording immediately after close command
            if contact_callback is not None:
                world.add_timestep_callback("record_contacts", contact_callback)

        elif phase[0] == 1 and elapsed[0] >= CONTACT_SETTLE_DELAY_S:
            world.remove_timestep_callback("grasp_close")
            _save_contacts()

    world.add_timestep_callback("grasp_close", _on_timestep)
    print(
        f"run_grasp_sim_omniverse: On Play, grippers close at t={GRASP_CLOSE_DELAY_S}s, "
        f"contacts sampled at t={GRASP_CLOSE_DELAY_S + CONTACT_SETTLE_DELAY_S}s."
    )


def main():
    """Entry when run as script. Assumes Isaac Sim has already loaded the stage."""
    run_grasp_on_play()


if __name__ == "__main__":
    main()
