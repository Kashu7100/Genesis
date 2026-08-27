"""An articulated arm simulated by the MochiSolver: a Franka Panda on the ground tracks joint targets with PD control.

Joint drives, joint limits, armature and damping enter the same implicit Newton solve as the rigid-body inertia and
the contact penalty, so stiff position gains remain stable at a 10 ms step.
"""

import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-n", "--n_steps", type=int, default=300, help="Number of simulation steps")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8),
        viewer_options=gs.options.ViewerOptions(camera_pos=(2.0, -2.0, 1.5), camera_lookat=(0.0, 0.0, 0.5)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Mochi.Rigid(),
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.06, 0.06, 0.06), pos=(0.65, 0.0, 0.03)),
        material=gs.materials.Mochi.Rigid(),
    )
    scene.build()

    franka.set_dofs_kp(np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0]))
    franka.set_dofs_kv(np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0]))
    targets = (
        np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8, 0.04, 0.04]),
        np.array([0.8, 0.2, 0.0, -1.6, 0.0, 1.8, 0.0, 0.0, 0.0]),
    )
    for i_step in range(args.n_steps):
        q_target = targets[(i_step // 150) % len(targets)]
        franka.control_dofs_position(q_target)
        scene.step()
        if i_step % 50 == 0:
            q_error = np.abs(np.asarray(franka.get_dofs_position().cpu()) - q_target).max()
            print(
                f"step {i_step:4d}: max joint error={q_error:.4f} rad, box z={box.get_pos()[2]:.4f}, "
                f"newton iterations={int(scene.mochi_solver.get_convergence_info()['n_iter'][0])}"
            )


if __name__ == "__main__":
    main()
