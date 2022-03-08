from wrapped_function import *
from t1005_graph import *

from wrapped_function import *
import sympy_function

_my_visual_studio_project_root = ""
UserLibrary.set_global_project_root(_my_visual_studio_project_root)

def test_case_1():
    g = Graph(name='Radius')
    x, y = g.state_inputs(['x', 'y'], 'double')
    c = g.config_inputs(['c'], 'UserType')
    radius = x ** 2 + y ** 3

    radius.set_name('r')

    create_rad = wrap_graph(g, [c, x, y], [radius], "CrtRad")

    create_rad.dump_to_lib(UserLibrary("generated", "test_case_1"))


test_case_1()
    # TODO(): test validity in C++ project. operators. Bp process for gradients. (single output)

