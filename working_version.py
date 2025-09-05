import gstaichi as ti
import genesis as gs
import numpy as np
import dataclasses

# We want our code to work for both cases: [IS_DIFFERENTIABLE = True] and [IS_DIFFERENTIABLE = False]
IS_DIFFERENTIABLE = True

"""
====================================================== COMMON CODE ======================================================
"""

ti.init(arch=ti.cuda)

USE_NDARRAY = not IS_DIFFERENTIABLE
V = ti.ndarray if USE_NDARRAY else ti.field
V_ANNOTATION = ti.types.ndarray() if USE_NDARRAY else ti.template()


@dataclasses.dataclass
class StructRigidGlobalInfo:
    n_links: V_ANNOTATION


MAX_N = 11
N1 = 3
N2 = 7


def get_rigid_global_info():
    kwargs = {"n_links": V(dtype=ti.i32, shape=(2,))}

    if USE_NDARRAY:
        return StructRigidGlobalInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassRigidGlobalInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassRigidGlobalInfo()


RigidGlobalInfo = StructRigidGlobalInfo if USE_NDARRAY else ti.template()
rigid_global_info = get_rigid_global_info()
rigid_global_info.n_links[0] = N1
rigid_global_info.n_links[1] = N2
array_a = V(dtype=ti.f32, shape=(MAX_N,), needs_grad=True if IS_DIFFERENTIABLE else False)

loss = ti.field(ti.f32, shape=(), needs_grad=True)

"""
====================================================== WORKING VERSION ======================================================
"""


#                               IS_DIFFERENTIABLE = False                     IS_DIFFERENTIABLE = True
# forward pass                    dynamic range over N                           dynamic range over N
# backward pass                             -                                  static range over MAX_N
@ti.kernel
def working_range(
    a: V_ANNOTATION,
    rigid_global_info: RigidGlobalInfo,
    max_n_links: ti.template(),
    enable_backward: ti.template(),
):
    for i in range(rigid_global_info.n_links.shape[0]):
        if ti.static(not enable_backward):
            for j in range(rigid_global_info.n_links[i]):
                loss[None] += a[j]
        else:
            for j in ti.static(range(max_n_links)):
                if j < rigid_global_info.n_links[i]:
                    loss[None] += a[j]


print(f"\nUsing IS_DIFFERENTIABLE = {IS_DIFFERENTIABLE}")
array_a.from_numpy(np.ones(MAX_N, dtype=np.float32))

if IS_DIFFERENTIABLE:
    loss.grad[None] = 1.0
    # In the forward pass, we can pass max_n_links = 0, as it is not used
    working_range(a=array_a, rigid_global_info=rigid_global_info, max_n_links=0, enable_backward=False)
    print("Forward result for working_range:")
    print(loss.to_numpy())
    # In the backward pass, we need to pass max_n_links = MAX_N, as it is used
    working_range.grad(a=array_a, rigid_global_info=rigid_global_info, max_n_links=MAX_N, enable_backward=True)
    print("Backward result for working_range:")
    print(array_a.grad.to_numpy())
else:
    working_range(a=array_a, rigid_global_info=rigid_global_info, max_n_links=0, enable_backward=False)
    print("Forward result for working_range:")
    print(loss.to_numpy())
    print("Does not support backward pass, since IS_DIFFERENTIABLE is False")
