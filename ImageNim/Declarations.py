# 2015.04.24 01:20:45 CDT
import cv2

BLUR_GAUSSIAN = 'GaussianBlur'
SOBEL = 'Sobel'
THRESHOLD_OTSU = 'OTSU'
LINEAR_HOUGH = 'HOUGH'
SQUARE_HOUGH = 'SQUARE'
THRESHOLD_ITERATIVE = 'ITER_THRE'
_OTSU = cv2.THRESH_BINARY + cv2.THRESH_OTSU