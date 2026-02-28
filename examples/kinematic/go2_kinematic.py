"""
Example: KinematicEntity as a ghost reference motion.

Creates a Go2 quadruped as a physics-simulated entity alongside a second Go2
loaded as a KinematicEntity (visualization-only ghost).  The kinematic entity follows a
simple sinusoidal joint trajectory while the physics robot is free to fall and
interact with the ground.  This demonstrates how KinematicEntity can display a
reference motion without affecting simulation speed or physics.
"""

import argparse
import math

import numpy as np
import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis")
    args = parser.parse_args()

    gs.init()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=args.vis,
    )

    # ── Ground plane ─────────────────────────────────────────────────
    scene.add_entity(gs.morphs.Plane())

    # ── Physics Go2 (normal rigid entity) ────────────────────────────
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.5, 0.42),
        ),
    )

    # ── Ghost Go2 (kinematic entity — visualization only) ─────────────
    ghost = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, -0.5, 0.42),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(color=(0.4, 0.7, 1.0), opacity=0.5),
    )

    scene.build()

    # ── Default standing pose (joint angles only, 12 DOFs) ───────────
    joint_angles = torch.tensor(
        [
            0.0,
            0.8,
            -1.5,  # FR: hip, thigh, calf
            0.0,
            0.8,
            -1.5,  # FL
            0.0,
            1.0,
            -1.5,  # RR
            0.0,
            1.0,
            -1.5,  # RL
        ],
        dtype=torch.float32,
        device=gs.device,
    )

    # Get joint DOF indices (skip the 6 base DOFs of the free joint)
    joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    robot_dof_idx = np.array([robot.get_joint(name).dofs_idx_local[0] for name in joint_names])
    ghost_dof_idx = np.array([ghost.get_joint(name).dofs_idx_local[0] for name in joint_names])

    robot.set_dofs_position(joint_angles, robot_dof_idx)
    ghost.set_dofs_position(joint_angles, ghost_dof_idx)

    for step in range(10000):
        t = step * scene.sim_options.dt

        # Sinusoidal reference trajectory for the ghost
        amplitude = 0.3
        freq = 2.0
        phase = 2.0 * math.pi * freq * t
        offset = amplitude * math.sin(phase)

        ref_angles = joint_angles.clone()
        # Oscillate thigh joints (indices 1, 4, 7, 10)
        ref_angles[1] += offset
        ref_angles[4] += offset
        ref_angles[7] -= offset
        ref_angles[10] -= offset

        ghost.set_dofs_position(ref_angles, ghost_dof_idx)

        scene.step()


if __name__ == "__main__":
    main()
