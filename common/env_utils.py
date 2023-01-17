import math
import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon
from geovoronoi import voronoi_regions_from_coords

from common.env import Custom_Environment


def make_env(domain, num_agents, step_length, max_steps, reward_type):
    def _init():
        env = Custom_Environment(domain, num_agents, step_length, max_steps, reward_type)
        return env

    return _init


def voronoi_based_area(num_agents: int, state: np.ndarray, domain: Polygon) -> np.ndarray:
    vehicle_position = state[0:num_agents, 0:2]

    region_polys, region_pts = voronoi_regions_from_coords(vehicle_position, domain)

    voronoi_region_area = np.zeros(num_agents, dtype=float)
    for key in region_polys:
        voronoi_region_area[key] = region_polys[key].area

    return voronoi_region_area


def calculate_potential(num_agents: int, state: np.ndarray, rd_distance: float) -> np.ndarray:
    potential = np.zeros(num_agents, dtype=float)
    for m in range(num_agents):

        p_m = state[m, 0:2]  # vehicle position.
        d = state[m, 6]  # domain distance.
        sign = state[m, 7]  # sign indicator

        # calculate the vehicle domain potential.
        if (sign * d) <= -0.5 * rd_distance:
            VH = 0
        else:
            VH = 0.5 * (((sign * d) + (0.5 * rd_distance)) ** 2)

        # calculate the inter-vehicle potential.
        VI = 0  # initializing inter-vehicle potential for vehicle m.
        for n in range(num_agents):
            if n != m:
                p_n = state[n, 0:2]
                p_mn = p_m - p_n
                norm = np.linalg.norm(p_mn, ord=2, keepdims=True)

                if norm < rd_distance:
                    VI_mn = (1 / 2) * (math.pow((norm - rd_distance), 2))
                else:
                    VI_mn = 0

                VI = VI + VI_mn

        potential[m] = (2 * VH + VI)

    return potential


def effective_coverage(num_agents: int, state: np.ndarray, rd_distance: int, domain: Polygon) -> float:
    p_i = state[0:2]
    s_i = Point(p_i[0], p_i[1]).buffer(rd_distance / 2)
    domain_intersection = domain.intersection(s_i)  # circular area intersect with domain part.

    # calculate the overlapping area with other agents.
    overlapping_area = 0
    pointer = 8  # relative vehicle position is start from 8-th column index in the state.
    for j in range(num_agents - 1):
        p_ij = state[pointer:pointer + 2]
        p_j = p_i - p_ij  # vector addition to recover the inter-vehicle vector.
        s_j = Point(p_j[0], p_j[1]).buffer(rd_distance / 2)
        pointer = pointer + 2

        overlapping_area = overlapping_area + domain_intersection.intersection(s_j).area

    """
    vehicle_position = state[0:num_agents, 0:2]
    
    effective_coverage_area = np.zeros(num_agents, dtype=float)
    for i in range(num_agents):
        p_i = vehicle_position[i]
        s_i = Point(p_i[0], p_i[1]).buffer(rd_distance/2)  # a circle area center at the vehicle_position.
        domain_intersection_i = domain.intersection(s_i)  # circular area intersect with domain part.

        # calculate the overlapping area with other agents.
        overlapping_area_i = 0
        cursor = 8  # relative vehicle position is start from 8-th column index in the state.
        for j in range(num_agents - 1):
            p_ij = state[i, cursor:cursor + 2]
            p_j = p_i + p_ij  # vector addition to recover the inter-vehicle vector.
            s_j = Point(p_j[0], p_j[1]).buffer(rd_distance/2)

            overlapping_area_i = overlapping_area_i + domain_intersection_i.intersection(s_j).area

        effective_coverage_area[i] = overlapping_area_i
    """

    return overlapping_area
