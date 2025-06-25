import numpy as np
import genesis as gs
import argparse

method = "mpr"
pile_type = "static"
precision = "32"
x_pos = 0.0

num_cubes = 5
batch_size = 16

args = argparse.ArgumentParser()
args.add_argument("--num_cubes", type=int, default=num_cubes, help="Number of cubes in the pyramid")
args.add_argument("--batch_size", type=int, default=batch_size, help="Number of environments in the batch")
args = args.parse_args()

num_cubes = args.num_cubes
batch_size = args.batch_size

print("Running with # cubes:", num_cubes, "and batch size:", batch_size)

gs.init(backend=gs.gpu, precision=precision)  # , logging_level="debug", precision="64")

# scene = gs.Scene(show_viewer=True)
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(x_pos, -3.5, 2.5),
        camera_lookat=(x_pos, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    rigid_options=gs.options.RigidOptions(
        box_box_detection=False,
        max_collision_pairs=1000,
        use_gjk_collision=(method == "gjk"),
        enable_mujoco_compatibility=False,
    ),
    show_viewer=False,
)

plane = scene.add_entity(gs.morphs.Plane(pos=(x_pos, 0, 0)))

# create pyramid of boxes
box_size = 0.25
if pile_type == "static":
    box_spacing = box_size
else:
    box_spacing = 1.1 * box_size
vec_one = np.array([1.0, 1.0, 1.0])
box_pos_offset = (x_pos - 0.5, 1, 0.0) + 0.5 * box_size * vec_one
boxes = {}
for i in range(num_cubes):
    for j in range(num_cubes - i):
        box = scene.add_entity(
            gs.morphs.Box(size=box_size * vec_one, pos=box_pos_offset + box_spacing * np.array([i + 0.5 * j, 0, j])),
        )

scene.build(n_envs=batch_size)
for i in range(3000):
    scene.step()
