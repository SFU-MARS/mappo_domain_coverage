import math
import numpy as np
from numpy import linalg
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import nearest_points


def rotate(pivot, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given pivot.
    """

    ox, oy = pivot
    px, py = point
    angle = math.radians(angle)

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([qx, qy])


class Pentagon(object):
    def __init__(self, center=np.array([0, 0]), offset=10):
        self.center = center
        self.center_distance = offset

        self.A = np.array([center[0], center[1] + self.center_distance])
        self.B = rotate(center, self.A, 360 / 5)
        self.C = rotate(center, self.B, 360 / 5)
        self.D = rotate(center, self.C, 360 / 5)
        self.E = rotate(center, self.D, 360 / 5)

        self.shape = Polygon([self.A, self.B, self.C, self.D, self.E])

        self.area = self.shape.area

    def measure_distance(self, point):

        x, y = point[0], point[1]
        point = Point(x, y)

        # vehicle is inside the domain.
        if self.shape.contains(point):

            sign = -1

            closest_point, _ = nearest_points(self.shape.boundary, point)
            c = np.array([closest_point.x, closest_point.y])
            h = sign * (c - np.array([x, y]))
            d = np.linalg.norm(h, ord=2)

            return h, d, sign, c

        # vehicle is outside the domain on on the boundary.
        else:

            sign = 1

            c, _ = nearest_points(self.shape, point)
            c = np.array([c.x, c.y])

            h = c - np.array([x, y])
            d = self.shape.distance(point)

            return h, d, sign, c
