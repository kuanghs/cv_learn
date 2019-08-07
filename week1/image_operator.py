#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


# In[ ]:


class ImgTransform(object):
    """image transform"""
    def __init__(self,img_path = './'):
        self.__filePath = img_path
        self.__image = cv2.imread(self.__filePath)
        print('image shape = {}'.format(self.__image.shape))
         
    #rect should be indicate region of img with two tuple, for example [(0,0), (100,100)] 
    def crop(self, rect):
        img_crop = self.__image[rect[0][0] : rect[1][0] , rect[0][1] : rect[1][1]]
        return img_crop

    def colorShift(self, bound = 50):
        channel_color = cv2.split(self.__image)
        colors = []
        for single in channel_color:
            rand_value = np.random.randint(-np.abs(bound), np.abs(bound))
            if rand_value == 0:
                pass
            elif rand_value < 0:
                lim = 0 - rand_value
                single[single < lim] = 0
                single[single >= lim] = (rand_value + single[single >= lim]).astype(self.__image.dtype)
            else:
                lim = 255 - rand_value
                single[single > lim] = 255
                single[single <= lim] = (rand_value + single[single <= lim]).astype(self.__image.dtype)
            colors.append(single)
        merge_img = cv2.merge(tuple(colors))
        return merge_img
    
    def imgRotation(self):
        rand_angle = np.random.randint(1,361)
        M = cv2.getRotationMatrix2D((self.__image.shape[1] / 2, self.__image.shape[0] / 2), rand_angle, 1)
        img_rotate = cv2.warpAffine(self.__image, M, (self.__image.shape[1], self.__image.shape[0]))
        return img_rotate
    
    def gammaAdjust(self, gamma = 1.0):
        if 0.0 == gamma:
            gamma = 1.0
        invGamma = 1.0 / gamma
        table = []
        for i in range(256):
            table.append(((i / 255) ** invGamma) * 255)
        table = np.array(table).astype(np.uint8)
        return cv2.LUT(self.__image, table)
    
    def randomWarp(self):
        random_margin = 60
        height, width, channels = self.__image.shape
        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(width - random_margin - 1, width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(width - random_margin - 1, width - 1)
        y3 = random.randint(height - random_margin - 1, height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(height - random_margin - 1, height - 1)

        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(width - random_margin - 1, width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(width - random_margin - 1, width - 1)
        dy3 = random.randint(height - random_margin - 1, height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(height - random_margin - 1, height - 1)
        
        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(self.__image, M , (width, height))
        return img_warp

      
    def imgShow(self, img = None):
        if img is None:
            img = self.__image
        cv2.imshow('Lena',img)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()

op = ImgTransform('E:/CV_LEARN/week1/lena/download.jpg')
#print(op.__doc__)
img = cv2.imread('E:/CV_LEARN/week1/lena/download.jpg')
cv2.imshow('img',img)
op.imgShow(op.gammaAdjust(2.0))