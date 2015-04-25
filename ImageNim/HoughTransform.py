# 2015.04.24 01:21:02 CDT
import numpy as np
import math
import itertools


class HoughTransform(object):

    def run(self, img, theta=1, rho_res=1):
        rhos, thetas, accumulator = self._hough_transform(img, 45, 3)
        parallel = {}
        lines = []
        for x in range(0, len(accumulator)):
            for y in range(0, len(accumulator[x])):
                rho = rhos[x]
                theta = thetas[y]
                if accumulator[x, y] > 65:
                    if theta not in parallel:
                        parallel[theta] = []
                    a = np.cos(theta * math.pi / 180.0)
                    b = np.sin(theta * math.pi / 180.0)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * -b)
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * -b)
                    y2 = int(y0 - 1000 * a)
                    parallel[theta].append([x1, y1, x2, y2])
                    lines.append([x1, y1, x2, y2])

        intersectionPoints = self._findIntersectionPoints(lines)
        squares = []
        for (p1, p2, p3, p4,) in self.window(intersectionPoints):
            if self.is_square(p1, p2, p3, p4):
                squares.append([p1, p2, p3, p4])

        return squares

    def _findIntersectionPoints(self, lines):
        intersectionPoints = []
        for x in range(0, len(lines)):
            for y in range(x + 1, len(lines)):
                x1, y1, x2, y2 = lines[x]
                x3, y3, x4, y4 = lines[y]
                d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if d != 0:
                    i, j = self.intersection((x1, y1, x2, y2),
                                             (x3, y3, x4, y4))
                    intersectionPoints.append((i, j))
        return intersectionPoints

    def _hough_transform(self, img, theta_res=1, rho_res=1):
        (nR, nC,) = img.shape
        theta = np.linspace(-90.0, 0.0, np.ceil(135.0 / theta_res) + 1.0)
        theta = np.concatenate((theta, -theta[(len(theta) - 2)::-1]))
        D = np.sqrt((nR - 1) ** 2 + (nC - 1) ** 2)
        q = np.ceil(D / rho_res)
        nrho = 2 * q + 1
        rho = np.linspace(-q * rho_res, q * rho_res, nrho)
        H = np.zeros((len(rho), len(theta)))
        for rowIdx in range(nR):
            for colIdx in range(nC):
                if img[rowIdx, colIdx]:
                    for thIdx in range(len(theta)):
                        rhoVal = colIdx * np.cos(theta[thIdx] * np.pi / 180.0) \
                            + rowIdx * np.sin(theta[thIdx] * np.pi / 180)
                        rhoIdx = np.nonzero(np.abs(rho - rhoVal) ==
                                            np.min(np.abs(rho - rhoVal)))[0]
                        H[(rhoIdx[0], thIdx)] += 1

        return (rho, theta, H)

    def intersection(self, a, b):
        (x1, y1, x2, y2,) = a
        (x3, y3, x4, y4,) = b
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if d != 0:
            x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2)
                 * (x3 * y4 - y3 * x4)) / d
            y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2)
                 * (x3 * y4 - y3 * x4)) / d
        else:
            x = -1
            y = -1
        return [x, y]

    def distsq(self, p1, p2):
        (x1, y1,) = p1
        (x2, y2,) = p2
        return (x2 - x1) ** 2 + (y2 - y1) ** 2

    def window(self, seq, n=4):
        """Returns a sliding window (of width n) over data from the iterable"""
        it = iter(seq)
        result = tuple(itertools.islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    def is_square(self, p1, p2, p3, p4):
        if any((a == b for (a, b,) in
                itertools.combinations((p1, p2, p3, p4), 2))):
            return False

        dis = [self.distsq(p1, p) for p in (p2, p3, p4)]
        pmin = min(dis)
        pmax = max(dis)

        if not (pmax == 2 * pmin and dis.count(pmin) == 2):
            return False

        far_pt = (p2, p3, p4)[dis.index(pmax)]
        return all((self.distsq(far_pt, p) == pmin
                    for p in (p2, p3, p4) if p != far_pt))
