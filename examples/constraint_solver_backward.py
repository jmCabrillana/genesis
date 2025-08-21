import genesis as gs
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

backend = gs.cpu
output_dir = "output/constraint_solver_backward"
os.makedirs(output_dir, exist_ok=True)

th.manual_seed(0)

### Set up scene
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
        constraint_solver=gs.constraint_solver.CG,
    ),
    show_viewer=False,
)

plane = scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0)))
box = scene.add_entity(gs.morphs.Box(size=(1, 1, 1), pos=(0, 0, 0.49)))

scene.build()
rigid_solver = scene._sim.rigid_solver
constraint_solver = rigid_solver.constraint_solver

# Step once to compute constraint solver's inputs: [mass], [jac], [aref], [efc_D], [force]. We compute gradients for them.
scene.step()
init_input_mass = th.from_numpy(rigid_solver._rigid_global_info.mass_mat.to_numpy())
init_input_jac = th.from_numpy(constraint_solver.constraint_state.jac.to_numpy())
init_input_aref = th.from_numpy(constraint_solver.constraint_state.aref.to_numpy())
init_input_efc_D = th.from_numpy(constraint_solver.constraint_state.efc_D.to_numpy())
init_input_force = th.from_numpy(rigid_solver.dofs_state.force.to_numpy())

# [acc_smooth] is dependent on [force], not an independent input ---> No need to compute gradient for it
init_input_acc_smooth = th.from_numpy(rigid_solver.dofs_state.acc_smooth.to_numpy())

# Initial output of the constraint solver
init_output_qacc = th.from_numpy(constraint_solver.qacc.to_numpy())
target_qacc = th.rand_like(init_output_qacc) * init_output_qacc.abs().mean()

# Number of constraints
n_constraints = th.from_numpy(constraint_solver.n_constraints.to_numpy())
print("Number of constraints: ", n_constraints[0].item())

# Set up optimization variables
ITER = 1000
lr = 1e3
input_mass = init_input_mass.clone()
input_jac = init_input_jac.clone()
input_aref = init_input_aref.clone()
input_efc_D = init_input_efc_D.clone()
input_force = init_input_force.clone()
input_acc_smooth = init_input_acc_smooth.clone()

input_mass.requires_grad = True
input_jac.requires_grad = True
input_aref.requires_grad = True
input_efc_D.requires_grad = True
input_force.requires_grad = True
optimizer = th.optim.Adam([input_mass, input_jac, input_aref, input_efc_D, input_force], lr=lr)

# Optimization
losses = []
bar = tqdm(range(ITER))
for i in bar:

    # Set input variables for the constraint solver
    with th.no_grad():
        curr_input_mass = input_mass.detach().clone().numpy()
        curr_input_jac = input_jac.detach().clone().numpy()
        curr_input_aref = input_aref.detach().clone().numpy()
        curr_input_efc_D = input_efc_D.detach().clone().numpy()
        curr_input_force = input_force.detach().clone().numpy()
        curr_input_acc_smooth = input_acc_smooth.detach().clone().numpy()

        rigid_solver._rigid_global_info.mass_mat.from_numpy(curr_input_mass)
        constraint_solver.constraint_state.jac.from_numpy(curr_input_jac)
        constraint_solver.constraint_state.aref.from_numpy(curr_input_aref)
        constraint_solver.constraint_state.efc_D.from_numpy(curr_input_efc_D)
        rigid_solver.dofs_state.force.from_numpy(curr_input_force)
        rigid_solver.dofs_state.acc_smooth.from_numpy(curr_input_acc_smooth)

    # Solve the constraint solver and get the output
    constraint_solver.resolve()
    output_qacc = constraint_solver.qacc.to_numpy()
    th_output_qacc = th.from_numpy(output_qacc).requires_grad_(True)

    # Compute loss and gradient of the output
    loss = (th_output_qacc - target_qacc).abs().mean()
    dL_dqacc = th.autograd.grad(loss, th_output_qacc)[0].numpy()

    # Compute gradients of the input variables: [mass], [jac], [aref], [efc_D], [force]
    constraint_solver.backward(dL_dqacc)

    # Fetch gradients of the input variables
    dL_dM = th.from_numpy(constraint_solver.constraint_state.dL_dM.to_numpy())
    dL_djac = th.from_numpy(constraint_solver.constraint_state.dL_djac.to_numpy())
    dL_daref = th.from_numpy(constraint_solver.constraint_state.dL_daref.to_numpy())
    dL_defc_D = th.from_numpy(constraint_solver.constraint_state.dL_defc_D.to_numpy())
    dL_dforce = th.from_numpy(constraint_solver.constraint_state.dL_dforce.to_numpy())

    # Set gradients of the input variables
    input_mass.grad = dL_dM
    input_jac.grad = dL_djac
    input_aref.grad = dL_daref
    input_efc_D.grad = dL_defc_D
    input_force.grad = dL_dforce

    # Update input variables
    optimizer.step()

    # Enforce constraints on the input variables
    with th.no_grad():
        input_mass.data = init_input_mass.data.clone()
        input_jac.data = init_input_jac.data.clone()
        input_aref.data = init_input_aref.data.clone()
        input_efc_D.data = init_input_efc_D.data.clone()

        # Recompute acc_smooth from the updated input variables
        updated_acc_smooth = th.linalg.solve(input_mass.squeeze(-1), input_force.squeeze(-1))
        input_acc_smooth.data = updated_acc_smooth.unsqueeze(-1)

    # Print loss
    bar.set_description(f"Loss: {loss.item():.20g}")
    losses.append(loss.item())

    # Save optimization graph
    if i % 10 == 0 or i == ITER - 1:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.savefig(f"{output_dir}/loss.png")
        plt.close()