# 2015.04.24 01:21:53 CDT
import cv2
import numpy as np


class IterativeThreshold(object):

    def run(self, img, delta=5):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        T1 = np.average(img)
        while True:
            (ret, tempImage,) = cv2.threshold(img, T1, 255, cv2.THRESH_BINARY)
            flattenedImage = tempImage.flatten()
            G1 = []
            G2 = []
            for x in flattenedImage:
                if x > T1:
                    G1.append(x)
                else:
                    G2.append(x)

            m1 = np.average(np.array(G1))
            m2 = np.average(np.array(G2))
            T2 = 0.5 * (m1 + m2)
            if abs(T2 - T1) < delta:
                return tempImage
            T1 = T2
