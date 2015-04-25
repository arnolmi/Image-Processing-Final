import cv2


class GaussianWrapper(object):
    def run(self, img, kernel=(5, 5), locality=0.0):
        return cv2.GaussianBlur(img, kernel, locality)
