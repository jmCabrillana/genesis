import argparse

import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            #res=(1280, 720),
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
            gravity=(0.0, 0.0, -1.0),
        ),
    )

    ########################## entities ##########################
    tank = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            euler=(90, 0, 0),
        ),
        #vis_mode="collision",
    )
    # plane = scene.add_entity(
    #     gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    #     # vis_mode="collision",
    # )
    ball = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.2,
            pos=(0.0, 0.0, 1),
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    # bottle = scene.add_entity(
    #     material=gs.materials.Rigid(rho=300),
    #     morph=gs.morphs.URDF(
    #         file="urdf/3763/mobility_vhacd.urdf",
    #         scale=0.09,
    #         pos=(0.0, 0.0, 1.0),
    #         euler=(0, 90, 0),
    #     ),
    #     vis_mode="collision",
    #     #visualize_contact=True
    # )

    ########################## build ##########################
    scene.build()
    for i in range(1000): #(135):
        # ball_pos = ball.get_pos()
        # print("Ball position:", ball_pos)
        scene.step()


if __name__ == "__main__":
    main()
