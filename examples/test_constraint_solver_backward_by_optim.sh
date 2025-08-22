python examples/test_constraint_solver_backward_by_optim.py --optimize_mass --lr=1e-4 --precision=64 --solver=Newton
python examples/test_constraint_solver_backward_by_optim.py --optimize_jac --lr=1e-4 --precision=64 --solver=Newton
python examples/test_constraint_solver_backward_by_optim.py --optimize_aref --lr=1e-4 --precision=64 --solver=Newton
python examples/test_constraint_solver_backward_by_optim.py --optimize_efc_D --lr=1e-4 --precision=64 --solver=Newton
python examples/test_constraint_solver_backward_by_optim.py --optimize_force --lr=1e-4 --precision=64 --solver=Newton
python examples/test_constraint_solver_backward_by_optim.py --optimize_mass --optimize_jac --optimize_aref --optimize_efc_D --optimize_force --lr=1e-4 --precision=64 --solver=Newton

python examples/test_constraint_solver_backward_by_optim.py --optimize_mass --lr=1e-4 --precision=32 --solver=Newton
python examples/test_constraint_solver_backward_by_optim.py --optimize_jac --lr=1e-4 --precision=32 --solver=Newton
python examples/test_constraint_solver_backward_by_optim.py --optimize_aref --lr=1e-4 --precision=32 --solver=Newton
python examples/test_constraint_solver_backward_by_optim.py --optimize_efc_D --lr=1e-4 --precision=32 --solver=Newton
python examples/test_constraint_solver_backward_by_optim.py --optimize_force --lr=1e-4 --precision=32 --solver=Newton
python examples/test_constraint_solver_backward_by_optim.py --optimize_mass --optimize_jac --optimize_aref --optimize_efc_D --optimize_force --lr=1e-4 --precision=32 --solver=Newton

python examples/test_constraint_solver_backward_by_optim.py --optimize_mass --lr=1e-4 --precision=64 --solver=CG
python examples/test_constraint_solver_backward_by_optim.py --optimize_jac --lr=1e-4 --precision=64 --solver=CG
python examples/test_constraint_solver_backward_by_optim.py --optimize_aref --lr=1e-4 --precision=64 --solver=CG
python examples/test_constraint_solver_backward_by_optim.py --optimize_efc_D --lr=1e-4 --precision=64 --solver=CG
python examples/test_constraint_solver_backward_by_optim.py --optimize_force --lr=1e-4 --precision=64 --solver=CG
python examples/test_constraint_solver_backward_by_optim.py --optimize_mass --optimize_jac --optimize_aref --optimize_efc_D --optimize_force --lr=1e-4 --precision=64 --solver=CG

python examples/test_constraint_solver_backward_by_optim.py --optimize_mass --lr=1e-4 --precision=32 --solver=CG
python examples/test_constraint_solver_backward_by_optim.py --optimize_jac --lr=1e-4 --precision=32 --solver=CG
python examples/test_constraint_solver_backward_by_optim.py --optimize_aref --lr=1e-4 --precision=32 --solver=CG
python examples/test_constraint_solver_backward_by_optim.py --optimize_efc_D --lr=1e-4 --precision=32 --solver=CG
python examples/test_constraint_solver_backward_by_optim.py --optimize_force --lr=1e-4 --precision=32 --solver=CG
python examples/test_constraint_solver_backward_by_optim.py --optimize_mass --optimize_jac --optimize_aref --optimize_efc_D --optimize_force --lr=1e-4 --precision=32 --solver=CG