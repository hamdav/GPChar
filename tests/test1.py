from gpchar.gpchar import GPChar
from gpchar.graphics import launch_dash_app

import numpy as np

def f_test(x):
    C0 = 195e12
    C1 = 4
    # a goes from 300 to 900: y0 varies by about the same
    # b has weak linear dependence and c has weak sqrt dependence
    y0 = C0*np.exp((x[0] - 600)/500) * (1 + 0.3*x[1]) * np.sqrt(x[2]+0.2)*1.5
    y1 = C1 * np.exp(np.cos((x[0]-600)/300)) * (1 + 0.1*x[1]) * np.sqrt(x[2])*2.5
    return np.array([y0, y1])


def test1():
    input_names = ["a", "b", "c"]
    output_names = ["f", "g"]
    bounds = [(300, 700), (0.1, 0.4), (0.5, 1.0)]
    #breakpoint()
    gpc = GPChar(f_test, bounds, 3, 2, "testdata.csv")
    #gpc.acquire_random_evaluations(2)

    launch_dash_app(gpc, bounds, input_names, output_names)


test1()
