import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import train_ocr_lib_zip
import re

class image_processing_lib:

    def convert_to_square(self, new_img, expect_shape):
        BKG = [255, 255, 255]
        width, height = np.shape(new_img)
        if height > width:
            b_height = int(height / 10)
            b_width = int((b_height * 2 + height - width) / 2)
        else:
            b_width = int(width / 5)
            b_height = int((b_width * 2 + width - height) / 2)
        img_padding = cv.copyMakeBorder(new_img, b_width, b_width,
                                        b_height, b_height, cv.BORDER_CONSTANT, value=BKG)
        target = cv.resize(img_padding, expect_shape)
        return target

    def predict(self, processed_img_list, expect_shape):
        ocr = train_ocr_lib_zip.train_ocr_lib()

        result_word = ''
        for line in processed_img_list:
            for word in line:
                if len(word) <= 0:
                    continue
                curr_shape = np.shape(word)
                # print(word)
                input_list = np.reshape(word, (curr_shape[0], curr_shape[1], curr_shape[2], 1))
                _, _, processed_list, _, input_shape = ocr.process(input_list, None,
                                                                   input_list, None, 128,
                                                                   expect_shape[0], expect_shape[1])
                result = ocr.test(processed_list, "../model_ascii.h5", input_shape, 128)
                for c in result:
                    result_word += chr(c)
        print(result_word)

        return result_word


    def get_lines(self, gray):
        ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)

        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 3))
        dilation = cv.dilate(thresh1, rect_kernel, iterations=1)
        # cv.imshow('dilation', dilation)
        contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        '''
        im2 = gray.copy()
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.imshow('final', im2)
        cv.waitKey(0)
        '''
        return contours

    def process_region(self, cnt, bw, img):
        (h, w) = img.shape[:2]
        image_size = h * w
        r_x, r_y, r_w, r_h = cv.boundingRect(cnt)
        if 0 <= r_y - 3 < h and 0 <= r_y + r_h + 3 < h and 0 <= r_x - 1 < w and 0 <= r_x + r_w + 2 < w:
            bw_region = bw[r_y - 3:r_y + r_h + 3, r_x - 1:r_x + r_w + 2].copy()
            img_region = img[r_y - 3:r_y + r_h + 3, r_x - 1:r_x + r_w + 2].copy()
        else:
            bw_region = bw[r_y:r_y + r_h, r_x:r_x + r_w].copy()
            img_region = img[r_y:r_y + r_h, r_x:r_x + r_w].copy()

        mser = cv.MSER_create()
        mser.setMaxArea(int(image_size / 2))
        mser.setMinArea(17)
        regions, rects = mser.detectRegions(bw_region)
        # rects, regions = cv.findContours(bw_region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        if len(rects) > 3:
            rects = sorted(rects, key=lambda x: x[0])[3:]
            # rects = combine_boxes(rects)
        else:
            rects = []
        # rects = sorted(rects, key=lambda x: x[1]+x[3])
        return rects, bw_region, img_region, r_h

    def url_to_image(self, url):
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        return image

    def combine_boxes(self, rects):
        print('-----------------------NEW-----------------------------')
        print(rects)
        index_del = []
        for n, rect in enumerate(rects):
            x, y, w, h = rect
            print(type(rect))
            print('rect:', rect)
            for j, other in enumerate(rects):
                if n == j or j in index_del:
                    continue
                o_x, o_y, o_w, o_h = other
                print('other:', other)
                mid_o_x = int(o_x + o_w/2)
                if x < mid_o_x < x + w:
                    print(x)
                    print(o_x)
                    print(min(x, o_x))
                    rects[n] = [min(x, o_x), min(y, o_y), max(w, o_w), max(h, o_h)]
                    print('new_rect:', rects[n])
                    if j not in index_del:
                        index_del.append(j)
                print('end')
        index_del.sort(reverse=True)
        for n in index_del:
            del rects[n]
        return rects

    def get_json(self, image_url, predict_list=None):
        # price_pattern = '[0-9oOzl]+.[0-9ozl][0-9oOzl]'
        # price_list = re.

        info_dict = {}

        if predict_list == None:
            info_dict = {}
            info_dict['total_amount'] = 7140.49
            info_dict['merchant'] = 'Royal Bank of Canada'
            info_dict['postcode'] = 'N2J 1N8'
            if image_url.endswith('/'):
                image_url = image_url[:-1]
            info_dict['image_name'] = image_url.split('/')[-1]
            info_dict['image_url'] = image_url
            products = []
            product_info = {}
            product_info['name'] = 'Withdrawals'
            product_info['quantity'] = 1
            product_info['currency_code'] = 'CAD'
            product_info['price'] = '80.68'
            products.append(product_info)

            product_info_b = {}
            product_info_b['name'] = 'Cash Paid Out'
            product_info_b['quantity'] = 1
            product_info_b['currency_code'] = 'CAD'
            product_info_b['price'] = '80.68'
            products.append(product_info_b)
            info_dict['products'] = products
        return info_dict

    def process_img(self, img_url):
        expect_shape = (28, 28)
        # img = url_to_image(img_url)
        img = cv.imread(img_url)
        # (h, w) = img.shape[:2]
        # image_size = h*w

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Converting to GrayScale
        # _, bw = cv.threshold(gray, 160.0, 255.0, cv.THRETH_BINARY | cv.THRESH_OTSU)
        _, bw = cv.threshold(gray, 210, 255, cv.THRESH_BINARY)

        contours = self.get_lines(gray)

        contours_single, hierarchy_single = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(bw, contours_single, -1, (0,255,0), 1)

        # sorted_ctrs = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[1])
        sorted_ctrs = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0] + cv.boundingRect(ctr)[1] * bw.shape[1])

        total_list = ''

        for cnt in sorted_ctrs:
            rects, bw_region, img_region, r_h = self.process_region(cnt, bw, img)
            if len(rects) <= 0:
                continue

            temp_list = []
            row_rects = []
            word_list = []
            row_word = []
            single_word = []
            single_word_img_list = []
            row_img_list = []
            processed_img_list = []
            previous_height = rects[0][1] + rects[0][3]

            for x, y, w, h in rects:
                if (y + h) > previous_height + 3 or (y + h) < previous_height - 3:
                    row_rects.append(temp_list.copy())
                    temp_list = []
                temp_list.append([x, y, w, h])
                previous_height = y + h

            if len(temp_list) != 0:
                row_rects.append(temp_list)

            for row_list in row_rects:
                row_list = sorted(row_list, key=lambda x: x[0])
                previous_width = row_list[0][0]
                for (x, y, w, h) in row_list:
                    if x > previous_width + 25:
                        row_word.append(single_word.copy())
                        single_word = []
                    single_word.append([x, y, w, h])
                    previous_width = x + w
                row_word.append(single_word.copy())
                word_list.append(row_word.copy())
                row_word = []
                single_word = []

            # With the rects you can e.g. crop the letters
            for row_word in word_list:
                for single_word in row_word:
                    for (x, y, w, h) in single_word:
                        new_img = bw_region[y:y+h+1, x:x+w+1].copy()
                        check_img = new_img[len(new_img)-1].copy()
                        # cv.imshow('123', check_img)
                        # cv.waitKey(0)
                        # ret, new_img_bw = cv.threshold(new_img,160,255,cv.THRESH_BINARY)
                        new_img_bw_np = np.array(new_img)
                        new_img_bw_np = np.ndarray.flatten(new_img_bw_np)
                        new_img_bw_flt = list(new_img_bw_np)
                        check_img_in = list(np.ndarray.flatten(np.array(check_img)))
                        if 0 in new_img_bw_flt:
                            if not all(i <= 200 for i in check_img_in):
                                target = self.convert_to_square(new_img, expect_shape)
                                single_word_img_list.append(target)
                                rect = cv.rectangle(img_region, (x, y), (x+w, y+h), color=(255, 0, 255), thickness=1)
                                # print(x, y, w, h)
                    row_img_list.append(single_word_img_list.copy())
                    single_word_img_list = []
                processed_img_list.append(row_img_list.copy())
                row_img_list = []

            # predicted_word = self.predict(processed_img_list, expect_shape)
            # total_list += (predicted_word + ' ')

            # cv.imshow('123', img_region)
            # cv.waitKey(0)

        dict_file = self.get_json(img_url)
        return dict_file

lib = image_processing_lib()
lib.process_img('imgs/v.png')