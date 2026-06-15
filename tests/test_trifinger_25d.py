"""
Minimal test script for the 25D TriFinger grasp state refactor.

Tests:
  1. components_to_grasp_state / grasp_state_to_components round-trip
  2. pose_9d → T_palm via rt_to_matrix (only first 9 dims)
  3. Shape assertions on q_pre, q_final, contact_heatmap
  4. GraspGenGenerator forward_train accepts 25D state when given q_pre / q_final
  5. GraspGenGenerator forward_inference returns T_palm, q_pre, q_final
"""

import torch

from grasp_gen.utils.math_utils import (
    components_to_grasp_state,
    grasp_state_to_components,
    rt_to_matrix,
)

NUM_JOINTS = 8
NUM_FINGERS = 3
BATCH = 4
NUM_PC_POINTS = 256
GRASP_REPR = "r3_6d"


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def random_pose_9d(B: int) -> torch.Tensor:
    """Random [B, 9] pose vector (xyz + rot6d)."""
    pose_9d = torch.randn(B, 9)
    return pose_9d


def random_q(B: int) -> tuple:
    """Random pre-grasp and final joint vectors."""
    q_pre = torch.randn(B, NUM_JOINTS)
    q_final = torch.randn(B, NUM_JOINTS)
    return q_pre, q_final


def random_contact_heatmap(N: int) -> torch.Tensor:
    """Random [N, num_fingers] contact heatmap (values in [0, 1])."""
    return torch.rand(N, NUM_FINGERS)


# ---------------------------------------------------------------------------
# Test 1: round-trip through helper functions
# ---------------------------------------------------------------------------


def test_grasp_state_round_trip():
    pose_9d = random_pose_9d(BATCH)
    q_pre, q_final = random_q(BATCH)

    state = components_to_grasp_state(pose_9d, q_pre, q_final)

    assert state.shape == (BATCH, 25), f"Expected (B, 25), got {state.shape}"

    pose_9d_back, q_pre_back, q_final_back = grasp_state_to_components(state)

    assert torch.allclose(pose_9d, pose_9d_back), "pose_9d mismatch after round-trip"
    assert torch.allclose(q_pre, q_pre_back), "q_pre mismatch after round-trip"
    assert torch.allclose(q_final, q_final_back), "q_final mismatch after round-trip"

    print("[PASS] test_grasp_state_round_trip")


# ---------------------------------------------------------------------------
# Test 2: pose_9d → T_palm
# ---------------------------------------------------------------------------


def test_rt_to_matrix_uses_only_first_9_dims():
    pose_9d = random_pose_9d(BATCH)
    q_pre, q_final = random_q(BATCH)
    state_25d = components_to_grasp_state(pose_9d, q_pre, q_final)

    # Must only pass the first 9 dims
    T_palm = rt_to_matrix(state_25d[:, :9], GRASP_REPR)

    assert T_palm.shape == (BATCH, 4, 4), f"Expected (B, 4, 4), got {T_palm.shape}"

    # Verify: same result when calling with pose_9d directly
    T_palm_direct = rt_to_matrix(pose_9d, GRASP_REPR)
    assert torch.allclose(T_palm, T_palm_direct, atol=1e-6), "T_palm mismatch"

    print("[PASS] test_rt_to_matrix_uses_only_first_9_dims")


# ---------------------------------------------------------------------------
# Test 3: shape assertions
# ---------------------------------------------------------------------------


def test_shape_assertions():
    pose_9d = random_pose_9d(BATCH)
    q_pre, q_final = random_q(BATCH)
    contact_heatmap = random_contact_heatmap(NUM_PC_POINTS)

    state = components_to_grasp_state(pose_9d, q_pre, q_final)

    p, qp, qf = grasp_state_to_components(state)
    assert p.shape == (BATCH, 9)
    assert qp.shape == (BATCH, NUM_JOINTS)
    assert qf.shape == (BATCH, NUM_JOINTS)
    assert contact_heatmap.shape == (NUM_PC_POINTS, NUM_FINGERS)

    print("[PASS] test_shape_assertions")


# ---------------------------------------------------------------------------
# Tests 4 & 5: GraspGenGenerator forward passes (require CUDA/ROCm GPU)
# ---------------------------------------------------------------------------


def _build_generator():
    from grasp_gen.models.generator import GraspGenGenerator

    gen = GraspGenGenerator(
        num_embed_dim=64,
        num_obs_dim=128,
        diffusion_embed_dim=128,
        num_diffusion_iters=10,
        num_diffusion_iters_eval=5,
        obs_backbone="pointnet",
        grasp_repr=GRASP_REPR,
        # "beta_schedule" is a config-template placeholder; use a real schedule for tests
        beta_schedule="squaredcos_cap_v2",
        gripper_name="franka_panda",
        pose_repr="mlp",
        num_grasps_per_object=4,
        num_joints=NUM_JOINTS,
    )
    return gen


def _build_batch(num_grasps: int = 4):
    """Build a minimal data dict for one object with num_grasps (GPU tensors)."""
    device = torch.device("cuda")
    pc = torch.randn(1, NUM_PC_POINTS, 3, device=device)

    # 4x4 grasp matrices
    grasps = torch.eye(4, device=device).unsqueeze(0).repeat(num_grasps, 1, 1)

    q_pre = torch.randn(1, num_grasps, NUM_JOINTS, device=device)
    q_final = torch.randn(1, num_grasps, NUM_JOINTS, device=device)

    return {
        "points": pc,
        "grasps": [grasps],  # list of [num_grasps, 4, 4]
        "q_pre": q_pre,
        "q_final": q_final,
    }


def test_generator_forward_train():
    gen = _build_generator().cuda()
    gen.eval()
    data = _build_batch(num_grasps=4)

    with torch.no_grad():
        outputs, losses, stats = gen.forward_train(data)

    # grasp_state_gt should be 25D
    assert "grasp_state_gt" in outputs
    gs = outputs["grasp_state_gt"]
    assert gs.shape[-1] == 9 + 2 * NUM_JOINTS, f"Expected 25D state, got {gs.shape}"

    print(f"[PASS] test_generator_forward_train  grasp_state_gt shape={gs.shape}")


# ---------------------------------------------------------------------------
# Test 5: GraspGenGenerator forward_inference returns T_palm, q_pre, q_final
# ---------------------------------------------------------------------------


def test_generator_forward_inference():
    gen = _build_generator().cuda()
    gen.eval()
    data = _build_batch(num_grasps=4)

    with torch.no_grad():
        outputs, _, _ = gen.forward_inference(data, return_metrics=False)

    assert "T_palm" in outputs, "Missing T_palm in outputs"
    assert "q_pre" in outputs, "Missing q_pre in outputs"
    assert "q_final" in outputs, "Missing q_final in outputs"
    assert "grasp_state" in outputs, "Missing grasp_state in outputs"

    T_palm = outputs["T_palm"]
    q_pre = outputs["q_pre"]
    q_final = outputs["q_final"]
    gs = outputs["grasp_state"]

    assert T_palm.shape[-2:] == (4, 4), f"Expected (..., 4, 4), got {T_palm.shape}"
    assert q_pre.shape[-1] == NUM_JOINTS, (
        f"Expected (..., {NUM_JOINTS}), got {q_pre.shape}"
    )
    assert q_final.shape[-1] == NUM_JOINTS, (
        f"Expected (..., {NUM_JOINTS}), got {q_final.shape}"
    )
    assert gs.shape[-1] == 9 + 2 * NUM_JOINTS, f"Expected 25D state, got {gs.shape}"

    print(
        f"[PASS] test_generator_forward_inference  "
        f"T_palm={T_palm.shape}  q_pre={q_pre.shape}  q_final={q_final.shape}"
    )


# ---------------------------------------------------------------------------
# Test 6: 25D state split correctness — joint dims must not bleed into pose
# ---------------------------------------------------------------------------


def test_joint_dims_do_not_affect_T_palm():
    """Changing joint dims must not change T_palm."""
    pose_9d = random_pose_9d(BATCH)
    q_pre, q_final = random_q(BATCH)

    state_a = components_to_grasp_state(pose_9d, q_pre, q_final)
    state_b = components_to_grasp_state(pose_9d, q_pre * 99, q_final * 99)

    T_palm_a = rt_to_matrix(state_a[:, :9], GRASP_REPR)
    T_palm_b = rt_to_matrix(state_b[:, :9], GRASP_REPR)

    assert torch.allclose(T_palm_a, T_palm_b, atol=1e-6), (
        "T_palm changed when only joint dims were modified — slice bug!"
    )

    print("[PASS] test_joint_dims_do_not_affect_T_palm")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_grasp_state_round_trip()
    test_rt_to_matrix_uses_only_first_9_dims()
    test_shape_assertions()
    test_generator_forward_train()
    test_generator_forward_inference()
    test_joint_dims_do_not_affect_T_palm()
    print("\nAll tests passed.")
