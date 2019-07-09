import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
og_img = cv.imread('a.png',0)
# img = cv.medianBlur(img,5)
ret,text_recognition_image = cv.threshold(og_img,160,255,cv.THRESH_BINARY)  # for text recognition
edge_detection_image = cv.adaptiveThreshold(og_img,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,15,2)  # for edge detection
# th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 150)',
            'Adaptive Mean Thresholding']
images = [og_img, text_recognition_image, edge_detection_image]

for i in range(3):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()