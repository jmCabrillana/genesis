import numpy as np
import genesis as gs
import argparse

backend = gs.cpu
gs.init(backend=backend, precision="32")

initial_contact = True

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, 0),
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, 2.5, 2.5),
        camera_lookat=(0.5, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    ),
    show_viewer=True,
)

# create two balls
ball1 = scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0, 0), radius=0.5), vis_mode="collision")
ball2 = scene.add_entity(
    gs.morphs.Sphere(pos=(0.9 if initial_contact else 1.1, 0, 0), radius=0.5), vis_mode="collision"
)

scene.build()

for i in range(10):
    print(f"========== Step {i} ==========")
    scene.step()

    # Get the contact data
    contacts = ball1.get_contacts(with_entity=ball2, exclude_self_contact=True)
    num_contacts = len(contacts["position"])

    if num_contacts > 0:
        print(f"Contact detected")
        for j in range(num_contacts):
            print(f"Contact {j}:")
            print(f"  Position: {contacts['position'][j]}")
            print(f"  Normal: {contacts['normal'][j]}")
            print(f"  Penetration: {contacts['penetration'][j]}")
    else:
        print(f"No contact detected, computing distance")
        # Compute minimum distance
        ball1_id = ball1._idx_in_solver
        ball2_id = ball2._idx_in_solver
        dist, w1, w2 = scene.rigid_solver.collider.compute_distance(ball1_id, ball2_id, 0)
        print(f"Minimum distance: {dist}")
        print(f"Witness point 1: {w1}")
        print(f"Witness point 2: {w2}")

    pass
