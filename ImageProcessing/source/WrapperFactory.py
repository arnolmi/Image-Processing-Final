# 2015.04.24 01:22:17 CDT
import cv2
from Declarations import *
from HoughTransform import HoughTransform
from Thresholding import IterativeThreshold

class GaussianWrapper(object):

    def run(self, img, kernel = (5, 5), locality = 0.0):
        return (cv2.GaussianBlur(img, kernel, locality), None)




class SobelWrapper(object):

    def run(self, img, type = cv2.CV_16S, size = 3):
        order = [(0, 1), (1, 0)]
        sobels = [ cv2.convertScaleAbs(cv2.Sobel(img, type, x, y, size)) for (x, y,) in order ]
        protoImg = cv2.addWeighted(sobels[0], 0.5, sobels[1], 0.5, 0)
        return (cv2.convertScaleAbs(protoImg), None)




class OtsuWrapper(object):

    def run(self, img):
        (ret, edges,) = ThresholdHelper().run(img, TypeDescriptors._OTSU)
        return (edges, None)




class HoughWrapper(object):

    def run(self, img, theta = 1, rho = 1):
        return (None, HoughTransform().run(img, theta, theta))




class ThresholdHelper(object):

    def run(self, img, method, _min = 0, _max = 255):
        return cv2.threshold(img, _min, _max, method)




class IterativeThresholdWrapper(object):

    def run(self, img):
        return (IterativeThreshold().run(img), None)



def operationFactory(typeDescriptor):
    if typeDescriptor == TypeDescriptors.BLUR_GAUSSIAN:
        return GaussianWrapper()
    if typeDescriptor == TypeDescriptors.SOBEL:
        return SobelWrapper()
    if typeDescriptor == TypeDescriptors.THRESHOLD_OTSU:
        return OtsuWrapper()
    if typeDescriptor == TypeDescriptors.LINEAR_HOUGH:
        return HoughWrapper()
    if typeDescriptor == TypeDescriptors.THRESHOLD_ITERATIVE:
        return IterativeThresholdWrapper()
    Assert(False)

