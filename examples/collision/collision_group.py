"""
NOTE: Geometries with the same non-zero collision group do not collide with each other.
This is useful to disable collision between parts of the same object.
This example demonstrates how to use the collision_group dictionary to disable self-collisions
on the go2 robot.
"""

import argparse

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init()

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # Define collision groups for the go2 robot.
    # We create two groups: one for the torso and one for the legs.
    # Collisions between links in the same group will be disabled.
    collision_group = {
        "torso": ["base"],
        "legs": [
            "FL_hip",
            "FL_thigh",
            "FL_calf",
            "FR_hip",
            "FR_thigh",
            "FR_calf",
            "RL_hip",
            "RL_thigh",
            "RL_calf",
            "RR_hip",
            "RR_thigh",
            "RR_calf",
        ],
    }

    # This go2 robot will not have self-collisions between the torso and legs,
    # because they are in different collision groups.
    scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.4),
            collision_group=collision_group,
        )
    )

    scene.build()

    for i in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
