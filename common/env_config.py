import numpy as np

from common.domain_square import Square
from common.domain_hexagon import Hexagon
from common.domain_pentagon import Pentagon
from common.domain_heptagon import Heptagon
from common.domain_polygon import ArbitraryPolygon

"""
  This file contains the default setup of environment used for training in this project
"""

# pentagon, polygon_1 and polygon_2 for 6 agents.
pentagon = {
    'domain': Pentagon(center=np.array([0, 0]), offset=4.5),
    'num_agents': 6,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}
polygon_1 = {
    'domain': ArbitraryPolygon(
        [np.array([-0.31, 6.47]), np.array([1.55, 4.93]), np.array([4.04, 4.67]), np.array([4.63, 2.99]),
         np.array([3.91, 1.36]), np.array([5.23, 0.56]), np.array([5.93, -0.74]), np.array([5.25, -1.95]),
         np.array([4.42, -2.72]), np.array([1.91, -2.15]), np.array([2.87, -3.52]), np.array([1.89, -3.73]),
         np.array([0.33, -5.43]), np.array([-1.54, -4.18]), np.array([-1.82, -2.47]), np.array([-0.76, -1.07]),
         np.array([-4.49, -2.76]), np.array([-5.31, -1.08]), np.array([-5.59, 0.16]), np.array([-4.44, 1.69]),
         np.array([-1.97, 0.99]), np.array([-2.09, 3.06]), np.array([-2.83, 4.74]), np.array([-1.54, 5.93])]),
    'num_agents': 6,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}
polygon_2 = {
    'domain': ArbitraryPolygon(
        [np.array([-0.19, 4.46]), np.array([1.50, 4.42]), np.array([2.79, 6.28]), np.array([5.34, 6.01]),
         np.array([5.87, 4.11]), np.array([5.39, 2.59]), np.array([4.47, 2.06]), np.array([5.42, 0.64]),
         np.array([5.41, -1.21]), np.array([3.72, -2.38]), np.array([2.29, -1.76]), np.array([1.10, -3.19]),
         np.array([-0.11, -3.89]), np.array([-1.71, -4.08]), np.array([-2.77, -5.14]), np.array([-4.05, -5.68]),
         np.array([-5.59, -4.84]), np.array([-6.14, -3.50]), np.array([-5.35, -1.35]), np.array([-6.07, -0.28]),
         np.array([-5.66, 1.76]), np.array([-3.53, 3.19]), np.array([-3.21956, 4.25]), np.array([-1.82, 4.90])]),
    'num_agents': 6,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}

# hexagon, polygon_3 and polygon_4 for 7 agents.
hexagon = {
    'domain': Hexagon(center=np.array([0, 0]), edge_length=5),
    'num_agents': 7,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}
polygon_3 = {
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
polygon_4 = {
    'domain': ArbitraryPolygon(
        [np.array([-5.12, 3.09]), np.array([-5.37, 5.35]), np.array([-3.23, 5.87]), np.array([-1.75,  6.92]),
         np.array([-0.53, 6.17]), np.array([1.06, 6.01]), np.array([2.13, 5.12]), np.array([3.01, 3.87]),
         np.array([4.16, 3.57]), np.array([2.48, 2.064]), np.array([4.05, 1.27]), np.array([4.68, -0.29]),
         np.array([4.13, -1.15]), np.array([6.64, -3.68]), np.array([3.74, -5.77]), np.array([1.65, -4.35]),
         np.array([-0.97, -4.82]), np.array([-3.26, -4.04]), np.array([-5.28, -4.67]), np.array([-5.59, -2.80]),
         np.array([-6.28, -1.35]), np.array([-4.82, -0.31]), np.array([-3.46,  0.07]), np.array([-3.21,  2.10])]),
    'num_agents': 7,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}

# heptagon, polygon_5 and polygon_6 for 8 agents.
heptagon = {
    'domain': Heptagon(center=np.array([0, 0]), offset=5),
    'num_agents': 8,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}
polygon_5 = {
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
polygon_6 = {
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

# square, polygon_7 and polygon_8 for 9 agents environment.
square = {
    'domain': Square(center=np.array([0, 0]), width=10),
    'num_agents': 9,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}
polygon_7 = {
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
polygon_8 = {
    'domain': ArbitraryPolygon(
        [np.array([3.43, 8.09]), np.array([6.03, 8.24]), np.array([6.90, 5.26]), np.array([8.04, 3.09]),
         np.array([7.79, 1.26]), np.array([6.31, 0.84]), np.array([4.13, 1.81]), np.array([3.49, 0.86]),
         np.array([2.06, 0.86]), np.array([1.04, -1.29]), np.array([1.37, -4.01]), np.array([-0.74, -4.80]),
         np.array([-2.81, -5.89]), np.array([-5.48, -5.90]), np.array([-6.85, -7.12]), np.array([-9.56, -5.96]),
         np.array([-9.03, -2.86]), np.array([-9.38, -0.21]), np.array([-6.37,  1.21]), np.array([-4.59,  1.06]),
         np.array([-2.37,  2.37]), np.array([0.49, 2.74]), np.array([0.69, 5.07]), np.array([2.54, 5.69])]),
    'num_agents': 9,
    'step_length': 0.1,
    'max_steps': 300,
    'reward_type': "LR",
}
