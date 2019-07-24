import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# # input original receipt image
# og_img = cv.imread('g.png',0)

# # binarization
# # for text recognition
# ret,text_recognition_image = cv.threshold(og_img,180,255,cv.THRESH_BINARY)

# # Gaussian Blurring
# blur = cv.GaussianBlur(og_img,(5,5),0)

# # Canny edge detection git
# edges = cv.Canny(blur, 100, 200)
# # Crop image based on Canny edge detection 
# edge_pts = np.argwhere(edges>0)
# y1,x1 = edge_pts.min(axis=0)
# y2,x2 = edge_pts.max(axis=0)
# crop_image = text_recognition_image[y1:y2, x1:x2]

# # new image title 
# titles = ['Original Image', 'Gaussian Blurring Image', 'Global Thresholding (v = 160)', 'Cropped Image']
# # image list 
# images = [og_img, blur, text_recognition_image, crop_image]

# # plot
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

img = cv.imread('i.png')
(h, w) = img.shape[:2]
image_size = h*w
mser = cv.MSER_create()
mser.setMaxArea(int(image_size/2))
mser.setMinArea(7)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Converting to GrayScale
_, bw = cv.threshold(gray, 0.0, 255.0, cv.THRESH_BINARY | cv.THRESH_OTSU)

regions, rects = mser.detectRegions(bw)

# print(regions)

# rects = sorted(rects, key=lambda x: x[0])
rects = sorted(rects, key=lambda x: x[1]+x[3])

img_list = []
temp_list = []
row_rects = []
word_list = []
row_word = []
single_word = []
single_word_img_list = []
row_img_list = []
processed_img_list = []
previous_height = rects[0][1] + rects[0][3]
previous_width = 0
row_counter = 0

for (x, y, w, h) in rects:
    if((y + h) > previous_height + 3):
        row_rects.append(temp_list.copy())
        temp_list = []
    temp_list.append([x, y, w, h])
    previous_height = y + h

for row_list in row_rects:
    row_list = sorted(row_list, key=lambda x: x[0])
    previous_width = row_list[0][0]
    for (x, y, w, h) in row_list:
        if(x > previous_width + 13):
            row_word.append(single_word.copy())
            single_word = []
        single_word.append([x, y, w, h])
        previous_width = x + w
    if (len(row_word) == 0):
        row_word.append(single_word.copy())
        word_list.append(row_word.copy())
        row_word = []
        single_word = []
    else:
        row_word.append(single_word.copy())
        word_list.append(row_word.copy())
        single_word = []
        row_word = []

# With the rects you can e.g. crop the letters
for row_word in word_list:
    for single_word in row_word:
        for (x, y, w, h) in single_word:
            new_img = gray[y:y+h+1, x:x+w+1].copy()
            new_img_in = img[y:y+h+1, x:x+w+1].copy()
            check_img = new_img_in[len(new_img_in)-1].copy()
            ret, new_img_bw = cv.threshold(new_img,80,255,cv.THRESH_BINARY)
            new_img_bw_np = np.array(new_img_bw)
            new_img_bw_np = np.ndarray.flatten(new_img_bw_np)
            new_img_bw_flt = list(new_img_bw_np)
            check_img_in = list(np.ndarray.flatten(np.array(check_img)))
            if 0 in new_img_bw_flt:
                if(not (all(i <= 200 for i in check_img_in))):
                    single_word_img_list.append(new_img_in)
                    rect = cv.rectangle(img, (x, y), (x+w, y+h), color=(255, 0, 255), thickness=1)
        row_img_list.append(single_word_img_list.copy())
        single_word_img_list = []
    processed_img_list.append(row_img_list.copy())
    row_img_list = []

# img_list.pop(0)

# img_list_row = []
# previous_hight = img_list[0][1] + img_list[0][3]
# column_counter = 0


# for (x, y, w, h) in img_list:
#     if((y + h) > previous_hight + 10):
#         column_counter = column_counter + 1
#     img_list_row[column_counter].append([x, y, w, h])

# print(img_list_row)

# new image title 
titles = []
# image list
images = []
# images.append(rect)

for row_img in processed_img_list:
    for single_img in row_img:
        for word_img in single_img:
            images.append(word_img)

print(processed_img_list[2])

titles.append('Original Image')

for x in range(len(images)):
    titles.append(str(x))
# plot

for i in range(50):
    plt.subplot(5,10,i+1),plt.imshow(images[i+149],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
