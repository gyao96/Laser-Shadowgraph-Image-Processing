#!/usr/bin/env python
# coding: utf-8
import cv2
import PIL
import numpy
def is_similar(img1, img2, method = 'BITWISE'):
    # compare bit by bit
    if method == 'BITWISE':
        # convert to grayscale if not
        if len(img1.shape) != 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) != 2:
            img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        # bitwise compare
        comp = img1 == img2
        [height, width] = comp.shape
        flag = True
        for i in range(height):
            for j in range(width):
                if comp[i][j]==False:
                    flag = False
        return flag
    # compare by distance hash
    if method == 'dHASH':
        hamming_distance = DHash.hamming_distance(img1, img2)
        if hamming_distance <= 0:
            return True
        else:
            return False

'''
Reference from
https://github.com/hjaurum/DHash
'''

class DHash(object):
    @staticmethod
    def calculate_hash(image):
        """
        计算图片的dHash值
        :param image: PIL.Image
        :return: dHash值,string类型
        """
        difference = DHash.__difference(image)
        # 转化为16进制(每个差值为一个bit,每8bit转为一个16进制)
        decimal_value = 0
        hash_string = ""
        for index, value in enumerate(difference):
            if value:  # value为0, 不用计算, 程序优化
                decimal_value += value * (2 ** (index % 8))
            if index % 8 == 7:  # 每8位的结束
                hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))  # 不足2位以0填充。0xf=>0x0f
                decimal_value = 0
        return hash_string

    @staticmethod
    def hamming_distance(first, second):
        """
        计算两张图片的汉明距离(基于dHash算法)
        :param first: Image或者dHash值(str)
        :param second: Image或者dHash值(str)
        :return: hamming distance. 值越大,说明两张图片差别越大,反之,则说明越相似
        """
        # A. dHash值计算汉明距离
        if isinstance(first, str):
            return DHash.__hamming_distance_with_hash(first, second)

        # B. image计算汉明距离
        hamming_distance = 0
        image1_difference = DHash.__difference(first)
        image2_difference = DHash.__difference(second)
        for index, img1_pix in enumerate(image1_difference):
            img2_pix = image2_difference[index]
            if img1_pix != img2_pix:
                hamming_distance += 1
        return hamming_distance

    @staticmethod
    def __difference(image):
        """
        *Private method*
        计算image的像素差值
        :param image: PIL.Image
        :return: 差值数组。0、1组成
        """
        resize_width = 9
        resize_height = 8
        # 1. resize to (9,8)
        smaller_image = cv2.resize(image,(resize_width,resize_height),interpolation=cv2.INTER_CUBIC)
        # 2. 灰度化 Grayscale
        if len(image.shape) != 2:
            grayscale_image = cv2.cvtColor(smaller_image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = smaller_image
        # 3. 比较相邻像素
        pixels = list(grayscale_image.flatten())
        difference = []
        for row in range(resize_height):
            row_start_index = row * resize_width
            for col in range(resize_width - 1):
                left_pixel_index = row_start_index + col
                difference.append(pixels[left_pixel_index] > pixels[left_pixel_index + 1])
        return difference

    @staticmethod
    def __hamming_distance_with_hash(dhash1, dhash2):
        """
        *Private method*
        根据dHash值计算hamming distance
        :param dhash1: str
        :param dhash2: str
        :return: 汉明距离(int)
        """
        difference = (int(dhash1, 16)) ^ (int(dhash2, 16))
        return bin(difference).count("1")
