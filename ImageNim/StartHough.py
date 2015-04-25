import cv2
from matplotlib import pyplot as plt
from ImageProcessor import ImageProcessor
import Declarations

s = ImageProcessor("/Volumes/Files/10Noise.png")

args = {'theta': 15, 'rho': 3}
procedure = [(Declarations.BLUR_GAUSSIAN, None),
             (Declarations.SOBEL, None),
             (Declarations.THRESHOLD_OTSU, None),
             (Declarations.LINEAR_HOUGH, args)]
s.generateBinaryImage(procedure)

for square in s.result:
    for point in square:
        cv2.circle(s.color, point, 2, (0, 255, 0), 2)

plt.subplot(3, 1, 1), plt.imshow(s.original, 'gray')
plt.title("stdev5"), plt.xticks([]), plt.yticks([])
plt.subplot(3, 1, 2), plt.imshow(s.working, 'gray')
plt.title("edges"), plt.xticks([]), plt.yticks([])
plt.subplot(3, 1, 3), plt.imshow(s.color, 'gray')
plt.title("Hough Transform"), plt.xticks([]), plt.yticks([])
plt.show()
