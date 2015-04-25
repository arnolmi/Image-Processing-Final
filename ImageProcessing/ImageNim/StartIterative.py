import cv2
import numpy as np
from matplotlib import pyplot as plt
from ImageProcessor import ImageProcessor
from Declarations import TypeDescriptors

s = ImageProcessor("/Volumes/Files/10Noise.png")

args = {'theta': 15, 'rho': 3}
procedure = [(TypeDescriptors.THRESHOLD_ITERATIVE, None)]
s.generateBinaryImage(procedure)
noise10 = s.original
result10 = s.working

s = ImageProcessor("/Volumes/Files/5Noise.png")

args = {'theta': 15, 'rho': 3}
procedure = [(TypeDescriptors.THRESHOLD_ITERATIVE, None)]
s.generateBinaryImage(procedure)
noise5 = s.original
result5 = s.working

images = [noise10, 0, result10,
          noise5, 0, result5,
          ]

titles = ['STDEV: 10 Noise','',"Iterative Thresholding",
          'STDEV: 5 Noise','',"Iterative Thresholding",
          ]
          
numimages = len(images)/3
for i in xrange(numimages):
    plt.subplot(numimages,3,i*3+1),plt.imshow(images[i*3], 'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(numimages,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(numimages,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])    
 

    
plt.show()