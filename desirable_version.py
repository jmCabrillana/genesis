import gstaichi as ti
import genesis as gs
import numpy as np
import dataclasses

# We want our code to work for both cases: [IS_DIFFERENTIABLE = True] and [IS_DIFFERENTIABLE = False]
IS_DIFFERENTIABLE = False

"""
====================================================== COMMON CODE ======================================================
"""

ti.init(arch=ti.cuda)

MAYBE_STATIC = ti.static if IS_DIFFERENTIABLE else gs._noop
USE_NDARRAY = not IS_DIFFERENTIABLE
V = ti.ndarray if USE_NDARRAY else ti.field
V_ANNOTATION = ti.types.ndarray() if USE_NDARRAY else ti.template()


@dataclasses.dataclass
class StructRigidGlobalInfo:
    n_links: V_ANNOTATION
    max_n_links: ti.i32


MAX_N = 11
N1 = 3
N2 = 7


def get_rigid_global_info():
    kwargs = {
        "n_links": V(dtype=ti.i32, shape=(2,)),
        "max_n_links": ti.i32,
    }

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
rigid_global_info.max_n_links = MAX_N
array_a = V(dtype=ti.f32, shape=(MAX_N,), needs_grad=True if IS_DIFFERENTIABLE else False)

loss = ti.field(ti.f32, shape=(), needs_grad=True)

"""
====================================================== DESIRABLE VERSION ======================================================
"""


#                               IS_DIFFERENTIABLE = False                     IS_DIFFERENTIABLE = True
# forward pass                  dynamic range over MAX_N                       static range over MAX_N
# backward pass                             -                                  static range over MAX_N
@ti.kernel
def desirable_range(a: V_ANNOTATION, rigid_global_info: RigidGlobalInfo):
    for i in range(rigid_global_info.n_links.shape[0]):
        for j in MAYBE_STATIC(range(rigid_global_info.max_n_links)):
            if j < rigid_global_info.n_links[i]:
                loss[None] += a[j]


print(f"\nUsing IS_DIFFERENTIABLE = {IS_DIFFERENTIABLE}")
array_a.from_numpy(np.ones(MAX_N, dtype=np.float32))

if IS_DIFFERENTIABLE:
    loss.grad[None] = 1.0
    desirable_range(a=array_a, rigid_global_info=rigid_global_info)
    print("Forward result for desirable_range:")
    print(loss.to_numpy())
    desirable_range.grad(a=array_a, rigid_global_info=rigid_global_info)
    print("Backward result for desirable_range:")
    print(array_a.grad.to_numpy())
else:
    try:
        desirable_range(a=array_a, rigid_global_info=rigid_global_info)
    except Exception as e:
        print("Expected exception, we want to solve this:")
        print(e)
        exit(1)
    print("Forward result for desirable_range:")
    print(loss.to_numpy())
    print("Does not support backward pass, since IS_DIFFERENTIABLE is False")
