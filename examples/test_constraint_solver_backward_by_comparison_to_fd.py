from sympy.printing.latex import true
import genesis as gs
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import argparse

backend = gs.cpu
output_dir = "output/constraint_solver_backward_fd"

os.makedirs(output_dir, exist_ok=True)

th.manual_seed(0)

### Set up scene
gs.init(backend=backend, precision="64")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    rigid_options=gs.options.RigidOptions(
        box_box_detection=False,
        max_collision_pairs=1000,
        use_gjk_collision=True,
        enable_mujoco_compatibility=False,
        constraint_solver=gs.constraint_solver.Newton,
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
target_qacc = th.rand_like(init_output_qacc) * init_output_qacc.abs().mean()

# Number of constraints
n_constraints = th.from_numpy(constraint_solver.n_constraints.to_numpy())
print("Number of constraints: ", n_constraints[0].item())

# Solve the constraint solver and get the output
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


def compute_loss(input_mass, input_jac, input_aref, input_efc_D, input_force):
    rigid_solver._rigid_global_info.mass_mat.from_numpy(input_mass.detach().clone().numpy())
    constraint_solver.constraint_state.jac.from_numpy(input_jac.detach().clone().numpy())
    constraint_solver.constraint_state.aref.from_numpy(input_aref.detach().clone().numpy())
    constraint_solver.constraint_state.efc_D.from_numpy(input_efc_D.detach().clone().numpy())
    rigid_solver.dofs_state.force.from_numpy(input_force.detach().clone().numpy())

    # Recompute acc_smooth from the updated input variables
    updated_acc_smooth = th.linalg.solve(input_mass.squeeze(-1), input_force.squeeze(-1))
    input_acc_smooth = updated_acc_smooth.unsqueeze(-1).detach().clone().numpy()

    rigid_solver.dofs_state.acc_smooth.from_numpy(input_acc_smooth)

    constraint_solver.resolve()
    output_qacc = constraint_solver.qacc.to_numpy()
    th_output_qacc = th.from_numpy(output_qacc).requires_grad_(True)
    loss = ((th_output_qacc - target_qacc) ** 2).mean()
    return loss


### Compute directional derivatives along random directions
FD_EPS = 1e-4
TRIALS = 100
errors = {}

# dL_dforce
dL_dforce_error = 0
for trial in range(TRIALS):
    rand_dforce = th.randn_like(init_input_force)
    rand_dforce = th.nn.functional.normalize(
        rand_dforce,
        dim=0,
    )

    dL = (rand_dforce * dL_dforce).sum()

    # 1 * eps
    input_force = init_input_force + rand_dforce * FD_EPS
    lossP1 = compute_loss(init_input_mass, init_input_jac, init_input_aref, init_input_efc_D, input_force)

    # -1 * eps
    input_force = init_input_force - rand_dforce * FD_EPS
    lossP2 = compute_loss(init_input_mass, init_input_jac, init_input_aref, init_input_efc_D, input_force)
    dL_fd = (lossP1 - lossP2) / (2 * FD_EPS)

    dL_error = (dL - dL_fd).abs() / max(dL.abs(), dL_fd.abs(), gs.EPS)
    dL_dforce_error += dL_error

dL_dforce_error /= TRIALS
print(f"[dL_dforce] FD_EPS: {FD_EPS}, error: {dL_dforce_error:.4g}")

errors["dL_dforce"] = dL_dforce_error.item()

# dL_daref
dL_daref_error = 0
for trial in range(TRIALS):
    rand_daref = th.randn_like(init_input_aref)
    rand_daref = th.nn.functional.normalize(
        rand_daref,
        dim=0,
    )

    dL = (rand_daref * dL_daref).sum()

    # 1 * eps
    input_aref = init_input_aref + rand_daref * FD_EPS
    lossP1 = compute_loss(init_input_mass, init_input_jac, input_aref, init_input_efc_D, init_input_force)

    # -1 * eps
    input_aref = init_input_aref - rand_daref * FD_EPS
    lossP2 = compute_loss(init_input_mass, init_input_jac, input_aref, init_input_efc_D, init_input_force)
    dL_fd = (lossP1 - lossP2) / (2 * FD_EPS)

    dL_error = (dL - dL_fd).abs() / max(dL.abs(), dL_fd.abs(), gs.EPS)
    dL_daref_error += dL_error

dL_daref_error /= TRIALS
print(f"[dL_daref] FD_EPS: {FD_EPS}, error: {dL_daref_error:.4g}")
errors["dL_daref"] = dL_daref_error.item()

# dL_defc_D
dL_defc_D_error = 0
for trial in range(TRIALS):
    rand_defc_D = th.randn_like(init_input_efc_D)
    rand_defc_D = th.nn.functional.normalize(
        rand_defc_D,
        dim=0,
    )

    dL = (rand_defc_D * dL_defc_D).sum()

    input_efc_D = init_input_efc_D + rand_defc_D * FD_EPS
    lossP1 = compute_loss(init_input_mass, init_input_jac, init_input_aref, input_efc_D, init_input_force)

    # -1 * eps
    input_efc_D = init_input_efc_D - rand_defc_D * FD_EPS
    lossP2 = compute_loss(init_input_mass, init_input_jac, init_input_aref, input_efc_D, init_input_force)
    dL_fd = (lossP1 - lossP2) / (2 * FD_EPS)

    dL_error = (dL - dL_fd).abs() / max(dL.abs(), dL_fd.abs(), gs.EPS)
    dL_defc_D_error += dL_error

dL_defc_D_error /= TRIALS
print(f"[dL_defc_D] FD_EPS: {FD_EPS}, error: {dL_defc_D_error:.4g}")
errors["dL_defc_D"] = dL_defc_D_error.item()

# dL_djac
dL_djac_error = 0
for trial in range(TRIALS):
    rand_djac = th.randn_like(init_input_jac)
    rand_djac = th.nn.functional.normalize(
        rand_djac,
        dim=0,
    )

    dL = (rand_djac * dL_djac).sum()

    input_jac = init_input_jac + rand_djac * FD_EPS
    lossP1 = compute_loss(init_input_mass, input_jac, init_input_aref, init_input_efc_D, init_input_force)

    # -1 * eps
    input_jac = init_input_jac - rand_djac * FD_EPS
    lossP2 = compute_loss(init_input_mass, input_jac, init_input_aref, init_input_efc_D, init_input_force)
    dL_fd = (lossP1 - lossP2) / (2 * FD_EPS)

    dL_error = (dL - dL_fd).abs() / max(dL.abs(), dL_fd.abs(), gs.EPS)
    dL_djac_error += dL_error

dL_djac_error /= TRIALS
print(f"[dL_djac] FD_EPS: {FD_EPS}, error: {dL_djac_error:.4g}")
errors["dL_djac"] = dL_djac_error.item()

# dL_dM
dL_dM_error = 0
for trial in range(TRIALS):
    rand_dM = th.rand_like(init_input_mass)
    rand_dM = th.nn.functional.normalize(rand_dM, dim=(0, 1))
    rand_dM = (rand_dM + rand_dM.transpose(0, 1)) * 0.5

    dL = (rand_dM * dL_dM).sum()

    input_mass = init_input_mass + rand_dM * FD_EPS
    lossP1 = compute_loss(input_mass, init_input_jac, init_input_aref, init_input_efc_D, init_input_force)

    # -1 * eps
    input_mass = init_input_mass - rand_dM * FD_EPS
    lossP2 = compute_loss(input_mass, init_input_jac, init_input_aref, init_input_efc_D, init_input_force)
    dL_fd = (lossP1 - lossP2) / (2 * FD_EPS)

    dL_error = (dL - dL_fd).abs() / max(dL.abs(), dL_fd.abs(), gs.EPS)
    dL_dM_error += dL_error

dL_dM_error /= TRIALS
print(f"[dL_dM] FD_EPS: {FD_EPS}, error: {dL_dM_error:.4g}")
errors["dL_dM"] = dL_dM_error.item()

# Save errors
import yaml

with open(os.path.join(output_dir, "relative_errors.yaml"), "w") as f:
    yaml.dump(errors, f)

# Print errors
for key, value in errors.items():
    print(f"  {key}: {value * 100:.4g}%")
print()
