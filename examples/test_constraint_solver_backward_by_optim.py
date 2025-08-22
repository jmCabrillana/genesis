from sympy.printing.latex import true
import genesis as gs
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--optimize_mass", action="store_true")
parser.add_argument("--optimize_jac", action="store_true")
parser.add_argument("--optimize_aref", action="store_true")
parser.add_argument("--optimize_efc_D", action="store_true")
parser.add_argument("--optimize_force", action="store_true")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--precision", type=str, default="64")
parser.add_argument("--solver", type=str, default="Newton")
args = parser.parse_args()

optimize_mass = args.optimize_mass
optimize_jac = args.optimize_jac
optimize_aref = args.optimize_aref
optimize_efc_D = args.optimize_efc_D
optimize_force = args.optimize_force

backend = gs.cpu
output_dir = "output/constraint_solver_backward_optimize"
output_dir = output_dir + f"_precision_{args.precision}"
output_dir = output_dir + f"_lr_{args.lr}"
if args.solver == "CG":
    output_dir = output_dir + "_CG"
elif args.solver == "Newton":
    output_dir = output_dir + "_Newton"

output_filename = ""
if optimize_mass:
    output_filename += "_mass"
if optimize_jac:
    output_filename += "_jac"
if optimize_aref:
    output_filename += "_aref"
if optimize_efc_D:
    output_filename += "_efcD"
if optimize_force:
    output_filename += "_force"

if not optimize_mass and not optimize_jac and not optimize_aref and not optimize_efc_D and not optimize_force:
    print("No optimization variables are selected. Exiting...")
    exit()

os.makedirs(output_dir, exist_ok=True)

th.manual_seed(0)

### Set up scene
gs.init(backend=backend, precision=str(args.precision))

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    rigid_options=gs.options.RigidOptions(
        box_box_detection=False,
        max_collision_pairs=1000,
        use_gjk_collision=True,
        enable_mujoco_compatibility=False,
        constraint_solver=gs.constraint_solver.CG if args.solver == "CG" else gs.constraint_solver.Newton,
    ),
    show_viewer=False,
)

plane = scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0)))
box = scene.add_entity(gs.morphs.Box(size=(1, 1, 1), pos=(10, 10, 0.49)))
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

scene.build()
rigid_solver = scene._sim.rigid_solver
constraint_solver = rigid_solver.constraint_solver

qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
franka.set_qpos(qpos)

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
target_qacc = th.sign(th.randn_like(init_output_qacc)) * 10 + init_output_qacc

# Number of constraints
n_constraints = th.from_numpy(constraint_solver.n_constraints.to_numpy())
print("Number of constraints: ", n_constraints[0].item())

# Set up optimization variables
ITER = 10000
lr = float(args.lr)

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
    loss = ((th_output_qacc - target_qacc) ** 2).mean()
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
        if not optimize_mass:
            input_mass.data = init_input_mass.data.clone()
        else:
            # Enforce mass matrix to be symmetric
            input_mass.data = (input_mass.data + input_mass.data.transpose(0, 1)) / 2.0

        if not optimize_jac:
            input_jac.data = init_input_jac.data.clone()

        if not optimize_aref:
            input_aref.data = init_input_aref.data.clone()

        if not optimize_efc_D:
            input_efc_D.data = init_input_efc_D.data.clone()
        else:
            input_efc_D.data[init_input_efc_D > 0.0] = th.clamp(
                input_efc_D.data[init_input_efc_D > 0.0], min=gs.EPS, max=1.0 / gs.EPS
            )

        if not optimize_force:
            input_force.data = init_input_force.data.clone()

        # Recompute acc_smooth from the updated input variables
        updated_acc_smooth = th.linalg.solve(input_mass.squeeze(-1), input_force.squeeze(-1))
        input_acc_smooth.data = updated_acc_smooth.unsqueeze(-1)

    # Print loss
    bar.set_description(f"Loss: {loss.item():.20g}")
    losses.append(loss.item())

    # Save optimization graph
    if i % 1000 == 0 or i == ITER - 1:
        # Create 2-panel graph: Left: plain loss, Right: log scale loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Left panel: Plain loss
        ax1.plot(range(len(losses)), losses, color="red", linewidth=2)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss", color="red")
        ax1.tick_params(axis="y", labelcolor="red")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Plain Loss")

        # Right panel: Log scale loss
        ax2.plot(range(len(losses)), np.log(losses), color="blue", linewidth=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Log Loss", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Log Scale Loss")

        # Add overall title
        title = "Variables: "
        if optimize_mass:
            title += "mass "
        if optimize_jac:
            title += "jac "
        if optimize_aref:
            title += "aref "
        if optimize_efc_D:
            title += "efc_D "
        if optimize_force:
            title += "force "
        title += f" | lr: {lr:.4g}"
        title += f" | precision: {args.precision}"
        title += f" | solver: {args.solver}"
        fig.suptitle(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{output_filename}.png", dpi=300, bbox_inches="tight")
        plt.close()
