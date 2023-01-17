import math
import numpy as np
from numpy import linalg
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import nearest_points


class Triangle(object):
    def __init__(self, center=np.array([0, 0]), edge_length=8):

        # initialize the parameters of the triangle class, use shapely package to calculate the distance measure.
        self.center = center
        self.edge_length = edge_length
        self.height = math.sqrt(math.pow(self.edge_length, 2) - math.pow(self.edge_length / 2, 2))
        self.area = 0.8 * self.edge_length * self.height

        self.A = np.array([center[0], center[1] + (self.height - ((1 / math.sqrt(3)) * (edge_length / 2)))])
        self.B = np.array([center[0] + edge_length / 2, center[1] - ((1 / math.sqrt(3)) * (edge_length / 2))])
        self.C = np.array([center[0] - edge_length / 2, center[1] - ((1 / math.sqrt(3)) * (edge_length / 2))])
        self.edge_AB = LineString([self.A, self.B])
        self.edge_BC = LineString([self.B, self.C])
        self.edge_CA = LineString([self.C, self.A])

        self.shape = Polygon([self.A, self.B, self.C])

    def measure_distance(self, point):
        x = point[0]
        y = point[1]

        point = Point(x, y)

        # vehicle is inside the domain.
        if self.shape.contains(point):

            sign = -1

            c1, _ = nearest_points(self.edge_AB, point)
            c2, _ = nearest_points(self.edge_BC, point)
            c3, _ = nearest_points(self.edge_CA, point)
            projection_point = [np.array([c1.x, c1.y]), np.array([c2.x, c2.y]), np.array([c3.x, c3.y])]

            h_vector = []
            for i in range(3):
                c = projection_point[i]
                h_vector.append(sign * (c - np.array([x, y])))

            norm = []
            for j in range(3):
                norm.append(linalg.norm(h_vector[j], ord=2, keepdims=True))

            min_value = min(norm)
            index = norm.index(min_value)
            h = h_vector[index]
            c = projection_point[index]
            d = self.shape.boundary.distance(point)

            return h, d, sign, c

        # vehicle is outside the domain on on the boundary.
        else:

            sign = 1

            c, _ = nearest_points(self.shape, point)
            c = np.array([c.x, c.y])

            h = c - np.array([x, y])
            d = self.shape.distance(point)

            return h, d, sign, c
