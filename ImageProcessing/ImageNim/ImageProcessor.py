# 2015.04.24 01:21:19 CDT
import cv2
import WrapperFactory
from Smoothing import *
from Declarations import *
from Thresholding import IterativeThreshold

class ThresholdHelper(object):

    def run(self, img, method, _min = 0, _max = 255):
        return cv2.threshold(img, _min, _max, method)

class ImageProcessor(object):

    def __init__(self, imageName):
        self.working = cv2.imread(imageName, 0)
        self.original = cv2.cvtColor(self.working, cv2.COLOR_GRAY2RGB)
        self.color = cv2.cvtColor(self.working, cv2.COLOR_GRAY2RGB)
        self.imageName = imageName

    def generateBinaryImage(self, procedure):
        """
            Currently Procedure doesn't hold arguments, but if needed put some arguments in the procedure List
        """
        for wrapper, args in procedure:
            working, result = (None, None)
            if args is None:
                working, result = WrapperFactory.operationFactory(wrapper).run(self.working)
            else:
                working, result = WrapperFactory.operationFactory(wrapper).run(self.working, **args)
            if working != None:
                self.working = working
            if result != None:
                self.result = result
