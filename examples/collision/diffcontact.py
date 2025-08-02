import numpy as np
import genesis as gs
import argparse
from scipy.spatial.transform import Rotation as R

backend = gs.cpu  # if cpu else gs.gpu

gs.init(backend=backend, precision="32")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    rigid_options=gs.options.RigidOptions(
        box_box_detection=False,
        max_collision_pairs=1000,
        use_gjk_collision=True,
        enable_mujoco_compatibility=False,
        use_diff_contact=False,  # Enable differentiable contact
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2, 2, 0.75),
        camera_lookat=(0.0, 0.0, 0.25),
        camera_fov=30,
        max_FPS=60,
    ),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0)))
# plane = scene.add_entity(
#     gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
# )

# create pyramid of boxes
box_size = 0.25
box_spacing = box_size
vec_one = np.array([1.0, 1.0, 1.0])
box_pos_offset = (0.0, 0.0, 0.0) + 0.5 * box_size * vec_one

box0 = scene.add_entity(
    gs.morphs.Box(size=box_size * vec_one, pos=box_pos_offset, fixed=True),
    visualize_contact=False,
    vis_mode="collision"
)
box1 = scene.add_entity(
    gs.morphs.Box(size=box_size * vec_one, pos=box_pos_offset + 0.8 * box_spacing * np.array([0, 0, 1]), fixed=True),
    visualize_contact=True,
    vis_mode="collision"
)

scene.build()

x_ang = 0.0
y_ang = 0.0
forward = 1.0
period = 100
for i in range(100000):

    x_ang += 0.03 * forward
    y_ang += 0.03 * forward

    box1.set_quat(R.from_euler("xy", [np.deg2rad(x_ang), np.deg2rad(y_ang)]).as_quat(scalar_first=True))
    # print("====================== Step:", i, "Angle:", ang, "======================")
    scene.step()

    if i == period:
        forward = -forward
    elif i > period and (i - period) % (period * 2) == 0:
        forward = -forward

    
    