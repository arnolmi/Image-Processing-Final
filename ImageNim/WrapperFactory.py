# 2015.04.24 01:22:17 CDT
import Declarations
from HoughTransform import HoughTransform
from Thresholding import IterativeThreshold
import cv2


class GaussianWrapper(object):
    def run(self, img, kernel=(5, 5), locality=0.0):
        return (cv2.GaussianBlur(img, kernel, locality), None)


class SobelWrapper(object):
    def run(self, img, type=cv2.CV_16S, size=3):
        order = [(0, 1), (1, 0)]
        sobels = [cv2.convertScaleAbs(cv2.Sobel(img, type, x, y, size))
                  for (x, y,) in order]
        protoImg = cv2.addWeighted(sobels[0], 0.5, sobels[1], 0.5, 0)
        return (cv2.convertScaleAbs(protoImg), None)


class OtsuWrapper(object):
    def run(self, img):
        ret, edges = ThresholdHelper().run(img, Declarations._OTSU)
        return (edges, None)


class HoughWrapper(object):
    def run(self, img, theta=1, rho=1):
        return (None, HoughTransform().run(img, theta, theta))


class ThresholdHelper(object):
    def run(self, img, method, _min=0, _max=255):
        return cv2.threshold(img, _min, _max, method)


class IterativeThresholdWrapper(object):
    def run(self, img):
        return (IterativeThreshold().run(img), None)


def operationFactory(typeDescriptor):
    if typeDescriptor == Declarations.BLUR_GAUSSIAN:
        return GaussianWrapper()
    if typeDescriptor == Declarations.SOBEL:
        return SobelWrapper()
    if typeDescriptor == Declarations.THRESHOLD_OTSU:
        return OtsuWrapper()
    if typeDescriptor == Declarations.LINEAR_HOUGH:
        return HoughWrapper()
    if typeDescriptor == Declarations.THRESHOLD_ITERATIVE:
        return IterativeThresholdWrapper()
    assert(False)
