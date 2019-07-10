import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# input original receipt image
og_img = cv.imread('a.png',0)

# binarization
# for text recognition
ret,text_recognition_image = cv.threshold(og_img,160,255,cv.THRESH_BINARY)

# Gaussian Blurring
blur = cv.GaussianBlur(og_img,(5,5),0)

# Canny edge detection 
edges = cv.Canny(blur, 100, 200)
# Crop image based on Canny edge detection 
edge_pts = np.argwhere(edges>0)
y1,x1 = edge_pts.min(axis=0)
y2,x2 = edge_pts.max(axis=0)
crop_image = text_recognition_image[y1:y2, x1:x2]

# new image title 
titles = ['Original Image', 'Gaussian Blurring Image', 'Global Thresholding (v = 160)', 'Cropped Image']
# image list 
images = [og_img, blur, text_recognition_image, crop_image]

# plot
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()