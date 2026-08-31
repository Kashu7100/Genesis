"""A Franka gripper closing on a soft tetrahedral cube.

This is the `soft_gripper` benchmark scene (the reinforcement-learning shaped acceptance scene of
`tests/mochi/benchmark`) run as a standalone demo: the hand descends onto a 5 cm neo-Hookean cube (~4000 tetrahedra)
under PD joint control while the fingertips press with 1 N each, squeezing the cube against the ground. The
articulated arm, the finite elements and the penalty contact enter one implicit Newton solve (mochi's default budget
of 4 iterations per step); the cube also acts as a collider, pushing the fingertip samples out through the signed
distance field of its deformed tetrahedra.
"""

import argparse

import numpy as np

import genesis as gs

FRANKA_START_QPOS = (-1.0418, 1.3805, 1.5885, -1.7021, -1.3798, 1.5783, 1.4467, 0.04, 0.04)
FRANKA_TARGET_QPOS = (-1.0095, 1.5617, 1.3595, -1.6830, -1.5855, 1.7817, 1.4595, 0.04, 0.04)
WORK_XY = (0.65, 0.0)
CUBE_SIZE = 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-n", "--n_steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=4),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.5, -0.8, 0.7), camera_lookat=(0.65, 0.0, 0.15)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Mochi.Rigid(friction=1.0, viscous_friction=1.0),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(CUBE_SIZE,) * 3,
            pos=(*WORK_XY, 0.5 * CUBE_SIZE),
            maxvolume=6e-8,
            nobisect=False,
        ),
        material=gs.materials.Mochi.Elastic(E=2e4, nu=0.4, rho=800.0, friction=1.0, viscous_friction=1.0),
        surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9)),
    )
    if args.record:
        cam = scene.add_camera(res=(640, 360), pos=(1.5, -0.8, 0.7), lookat=(0.65, 0.0, 0.15))
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_qpos(np.array(FRANKA_START_QPOS))
    franka.control_dofs_position(np.array(FRANKA_TARGET_QPOS)[motors_dof], motors_dof)
    franka.control_dofs_force(np.full(2, -1.0), fingers_dof)

    if args.record:
        cam.start_recording(save_to_filename="soft_gripper.mp4", fps=30)

    for i_step in range(args.n_steps):
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            cube_z = float(cube.get_vertices_position()[:, 2].max())
            hand_z = float(franka.get_link("hand").get_pos()[2])
            print(
                f"step {i_step:4d}: cube zmax={cube_z:.4f} hand z={hand_z:.4f} "
                f"newton iterations={int(info['n_iter'][0])}"
            )
        if args.record:
            cam.render()
    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
