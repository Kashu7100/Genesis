"""
NOTE: Geometries with the same non-zero collision group do not collide with each other.
This is useful to disable collision between parts of the same object.
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

    # This box will not collide with the green box because they are in the same collision group.
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0, 0.5),
            quat=(0, 0, 0, 1),
            size=(0.1, 0.1, 0.1),
            group=1,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    # This box will fall through the red box.
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0, 1.0),
            quat=(0, 0, 0, 1),
            size=(0.1, 0.1, 0.1),
            group=1,
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )

    scene.build()

    for i in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
