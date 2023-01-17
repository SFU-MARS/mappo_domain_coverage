import numpy as np

from common.domain_square import Square
from common.domain_hexagon import Hexagon
from common.domain_polygon import ArbitraryPolygon

"""
  This file contains the default setup of environment used for training in this project
"""

hexagon = {
    'domain': Hexagon(center=np.array([0, 0]), edge_length=5),
    'num_agents': 7,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}
polygon_1 = {
    'domain': ArbitraryPolygon(
        [np.array([1.23, 7.92]), np.array([3.84, 7.73]), np.array([5.66, 6.05]), np.array([7.55, 6.17]),
         np.array([9.47, 4.94]), np.array([10.13, 3.35]), np.array([9.47, 1.65]), np.array([7.67, -0.08]),
         np.array([5.93, -0.81]), np.array([4.67, -3.06]), np.array([5.39, -4.85]), np.array([5.17, -6.72]),
         np.array([3.56, -8.29]), np.array([1.29, -8.16]), np.array([-1.66, -7.56]), np.array([-3.55, -6.53]),
         np.array([-5.36, -4.37]), np.array([-7.12, -2.98]), np.array([-9.09, -2.01]), np.array([-9.63, 0.05]),
         np.array([-8.77, 2.33]), np.array([-6.17, 2.56]), np.array([-3.19, 3.58]), np.array([-1.59, 6.125])]),
    'num_agents': 7,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}

# polygon_2 and polygon_3 for 8 agents environment.
polygon_2 = {
    'domain': ArbitraryPolygon(
        [np.array([-0.58, 6.52]), np.array([3.61, 6.38]), np.array([7.22, 4.44]), np.array([8.7, 1.37]),
         np.array([10.71, -0.54]), np.array([11.15, -3.24]), np.array([9.18, -5.31]), np.array([6.31, -6.04]),
         np.array([4.15, -6.26]), np.array([2.08, -5.67]), np.array([1.01, -4.41]), np.array([-0.15, -3.76]),
         np.array([-1.60, -4.08]), np.array([-2.55, -4.84]), np.array([-2.85, -6.22]), np.array([-3.72, -7.56]),
         np.array([-5.93, -7.77]), np.array([-7.69, -6.87]), np.array([-8.87, -5.38]), np.array([-9.87, -3.33]),
         np.array([-10.30, -1.49]), np.array([-9.95, 0.61]), np.array([-8.71, 3.24]), np.array([-5.59, 5.34])]),
    'num_agents': 8,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}
polygon_3 = {
    'domain': ArbitraryPolygon(
        [np.array([0.76, 8.66]), np.array([3.42, 5.71]), np.array([3.02, 3.85]), np.array([4.78, 3.27]),
         np.array([6.92, 3.06]), np.array([9.07, 1.79]), np.array([10.76, 0.51]), np.array([9.50, -2.73]),
         np.array([7.24, -4.01]), np.array([6.01, -6.81]), np.array([3.16, -7.44]), np.array([1.66, -6.29]),
         np.array([0.44, -7.87]), np.array([-2.81, -7.36]), np.array([-5.38, -6.07]), np.array([-8.46, -5.31]),
         np.array([-9.08, -1.16]), np.array([-6.96, -0.15]), np.array([-5.59, 1.66]), np.array([-6.51, 3.22]),
         np.array([-8.29, 4.37]), np.array([-6.62, 7.21]), np.array([-4.88, 8.26]), np.array([-2.35, 7.31])]),
    'num_agents': 8,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}

# square and polygon_4 for 9 agents environment.
square = {
    'domain': Square(center=np.array([0, 0]), width=10),
    'num_agents': 9,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}
polygon_4 = {
    'domain': ArbitraryPolygon(
        [np.array([-0.38, 8.97]), np.array([3.21, 8.22]), np.array([4.69, 4.76]), np.array([6.13, 3.20]),
         np.array([9.50, 1.81]), np.array([10.31, -0.95]), np.array([9.27, -3.84]), np.array([7.75, -5.81]),
         np.array([5.01, -6.46]), np.array([1.79, -6.34]), np.array([-0.06, -5.25]), np.array([-1.84, -5.48]),
         np.array([-3.65, -8.45]), np.array([-6.07, -8.75]), np.array([-8.30, -7.35]), np.array([-8.67, -4.75]),
         np.array([-10.45, -3.24]), np.array([-9.46, -0.93]), np.array([-11.07, 1.57]), np.array([-10.31, 4.78]),
         np.array([-8.98, 6.54]), np.array([-6.64, 6.78]), np.array([-4.36, 7.47]), np.array([-2.21, 7.17])]),
    'num_agents': 9,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}
