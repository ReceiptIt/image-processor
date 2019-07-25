import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import train_ocr_lib_zip

class image_processing:

    def process_img(self, image_url):
        # expect_shape = (28, 28)
        # img = cv.imread('i.png')
        # (h, w) = img.shape[:2]
        # image_size = h*w
        # mser = cv.MSER_create()
        # mser.setMaxArea(int(image_size/2))
        # mser.setMinArea(7)

        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Converting to GrayScale
        # _, bw = cv.threshold(gray, 0.0, 255.0, cv.THRESH_BINARY | cv.THRESH_OTSU)

        # regions, rects = mser.detectRegions(bw)

        # # print(regions)

        # # rects = sorted(rects, key=lambda x: x[0])
        # rects = sorted(rects, key=lambda x: x[1]+x[3])

        # ocr = train_ocr_lib_zip.train_ocr_lib()

        # img_list = []
        # temp_list = []
        # row_rects = []
        # word_list = []
        # row_word = []
        # single_word = []
        # single_word_img_list = []
        # row_img_list = []
        # processed_img_list = []
        # previous_height = rects[0][1] + rects[0][3]
        # previous_width = 0
        # row_counter = 0

        # for (x, y, w, h) in rects:
        #     if((y + h) > previous_height + 3):
        #         row_rects.append(temp_list.copy())
        #         temp_list = []
        #     temp_list.append([x, y, w, h])
        #     previous_height = y + h

        # for row_list in row_rects:
        #     row_list = sorted(row_list, key=lambda x: x[0])
        #     previous_width = row_list[0][0]
        #     for (x, y, w, h) in row_list:
        #         if(x > previous_width + 13):
        #             row_word.append(single_word.copy())
        #             single_word = []
        #         single_word.append([x, y, w, h])
        #         previous_width = x + w
        #     if (len(row_word) == 0):
        #         row_word.append(single_word.copy())
        #         word_list.append(row_word.copy())
        #         row_word = []
        #         single_word = []
        #     else:
        #         row_word.append(single_word.copy())
        #         word_list.append(row_word.copy())
        #         single_word = []
        #         row_word = []

        # # With the rects you can e.g. crop the letters
        # for row_word in word_list:
        #     for single_word in row_word:
        #         for (x, y, w, h) in single_word:
        #             new_img = gray[y:y+h+1, x:x+w+1].copy()
        #             new_img_in = img[y:y+h+1, x:x+w+1].copy()
        #             check_img = new_img_in[len(new_img_in)-1].copy()
        #             ret, new_img_bw = cv.threshold(new_img,80,255,cv.THRESH_BINARY)
        #             new_img_bw_np = np.array(new_img_bw)
        #             new_img_bw_np = np.ndarray.flatten(new_img_bw_np)
        #             new_img_bw_flt = list(new_img_bw_np)
        #             check_img_in = list(np.ndarray.flatten(np.array(check_img)))
        #             if 0 in new_img_bw_flt:
        #                 if not all(i <= 200 for i in check_img_in):
        #                     BKG = [255, 255, 255]
        #                     width, height = np.shape(new_img_bw)
        #                     if height > width:
        #                         b_height = int(height / 5)
        #                         b_width = int((b_height * 2 + height - width) / 2)
        #                     else:
        #                         b_width = int(width / 5)
        #                         b_height = int((b_width * 2 + width - height) / 2)
        #                     img_padding = cv.copyMakeBorder(new_img_bw, b_width, b_width,
        #                                                     b_height, b_height, cv.BORDER_CONSTANT, value=BKG)
        #                     target = cv.resize(img_padding, expect_shape)
        #                     single_word_img_list.append(target)
        #                     rect = cv.rectangle(img, (x, y), (x+w, y+h), color=(255, 0, 255), thickness=1)
        #         row_img_list.append(single_word_img_list.copy())
        #         single_word_img_list = []
        #     processed_img_list.append(row_img_list.copy())
        #     row_img_list = []

        # # for img in img_list:
        # for line in processed_img_list:
        #     for word in line:
        #         if len(word) <= 0:
        #             continue
        #         curr_shape = np.shape(word)
        #         # print(word)
        #         input_list = np.reshape(word, (curr_shape[0], curr_shape[1], curr_shape[2], 1))
        #         _, _, processed_list, _, input_shape = ocr.process(input_list, None,
        #                                                            input_list, None, 128,
        #                                                            expect_shape[0], expect_shape[1])
        #         result = ocr.test(processed_list, "../model_digit_letter.h5", input_shape, 128)
        #         result_word = ''
        #         for c in result:
        #             result_word += chr(c)
        #         print(result_word)
        products_list = ['123']

        info_dict = {}
        info_dict['total_amount'] = 100.0
        info_dict['merchant'] = 'Jimmy Restaurant'
        info_dict['postcode'] = '43553'
        if image_url.endswith('/'):
            image_url = image_url[:-1]
        info_dict['image_name'] = image_url.split('/')[:-1]
        info_dict['image_url'] = image_url
        products = []
        for _ in range(1, 10):
            product_info = {}
            product_info['name'] = '123'
            product_info['description'] = '123'
            product_info['quantity'] = 1
            product_info['currency_code'] = 'CAD'
            product_info['price'] = '6.00'
            products.append(product_info)
        info_dict['products'] = products
        return info_dict


# if __name__ == '__main__':

#     ip = image_processing()
#     dict = ip.process_img('')
#     print(dict)

#     # new image title
#     titles = []
#     # image list
#     images = []
#     # images.append(rect)

#     for row_img in processed_img_list:
#         for single_img in row_img:
#             for word_img in single_img:
#                 images.append(word_img)

#     titles.append('Original Image')

#     for x in range(len(images)):
#         titles.append(str(x))
#     # plot

#     for i in range(50):
#         plt.subplot(5,10,i+1),plt.imshow(images[i],'gray')
#         plt.title(titles[i])
#         plt.xticks([]),plt.yticks([])
#     plt.show()
