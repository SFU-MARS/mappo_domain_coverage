import numpy as np
from shapely.geometry import Polygon


class Square(object):
    def __init__(self, center=np.array([0, 0]), width=10):
        self.center = center
        self.width = width
        self.area = width * width

        # using shapely Polygon class to define a square, for consistent purpose, not used for any calculation at here.
        self.A = np.array([center[0] + self.width / 2, center[1] + self.width / 2])
        self.B = np.array([center[0] - self.width / 2, center[1] + self.width / 2])
        self.C = np.array([center[0] - self.width / 2, center[1] - self.width / 2])
        self.D = np.array([center[0] + self.width / 2, center[1] - self.width / 2])
        self.shape = Polygon([self.A, self.B, self.C, self.D])

    def measure_distance(self, point):
        x = point[0]
        y = point[1]
        v = (self.width / 2)

        # vehicle is outside or on the domain.
        if abs(x) >= (self.width / 2) or abs(y) >= (self.width / 2):
            sign = 1

            if x >= v and (-v <= y <= v):
                c = np.array([v, y])
                h = c - point
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c

            elif x >= v and y >= v:
                c = np.array([v, v])
                h = c - point
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c

            elif y >= v and (-1 * v <= x <= v):
                c = np.array([x, v])
                h = c - point
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c

            elif x <= -1 * v and y >= v:
                c = np.array([-1 * v, v])
                h = c - point
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c

            elif x <= -1 * v and (-1 * v <= y <= v):
                c = np.array([-1 * v, y])
                h = c - point
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c

            elif x <= -1 * v and y <= -1 * v:
                c = np.array([-1 * v, -1 * v])
                h = c - point
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c

            elif y <= -1 * v and (-1 * v <= x <= v):
                c = np.array([x, -1 * v])
                h = c - point
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c

            elif x >= v and y <= -1 * v:
                c = np.array([v, -1 * v])
                h = c - point
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c

        # vehicle is inside the domain.
        elif abs(x) < v and abs(y) < v:
            sign = -1

            if (x > 0) and (y >= 0):
                a = np.array([v - x, v - y])
                index = np.argmin(a)

                if index == 0:
                    c = np.array([v, y])
                else:
                    c = np.array([x, v])
                h = sign * (c - point)
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c

            elif (x <= 0) and (y > 0):
                a = np.array([abs((-1 * v) - x), v - y])
                index = np.argmin(a)

                if index == 0:
                    c = np.array([-1 * v, y])
                else:
                    c = np.array([x, v])
                h = sign * (c - point)
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c

            elif (x < 0) and (y <= 0):
                a = np.array([abs((-1 * v) - x), abs((-1 * v) - y)])
                index = np.argmin(a)

                if index == 0:
                    c = np.array([-1 * v, y])
                else:
                    c = np.array([x, -1 * v])
                h = sign * (c - point)
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c

            elif (x >= 0) and (y < 0):
                a = np.array([v - x, abs((-1 * v) - y)])
                index = np.argmin(a)

                if index == 0:
                    c = np.array([v, y])
                else:
                    c = np.array([x, -1 * v])
                h = sign * (c - point)
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c
            else:
                c = np.array([0, v])
                h = sign * (c - point)
                d = np.linalg.norm(h, ord=2, keepdims=True)
                return h, d, sign, c
