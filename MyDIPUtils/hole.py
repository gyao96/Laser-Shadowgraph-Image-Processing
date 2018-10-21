#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
from MyDIPUtils.bitop import is_similar

def floodfill(image, FULL = False):

    im_in = image.copy()
    
    # Copy the thresholded image.
    im_floodfill = im_in.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_rawout = im_in | im_floodfill_inv
    im_rawout = cv2.dilate(im_rawout, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations = 4)
    im_rawout = cv2.erode(im_rawout, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations = 4)
    im_out = cv2.medianBlur(im_rawout,5)

    if FULL == True:
        while True:
            im_in = im_out.copy()
            
            # Copy the thresholded image.
            im_floodfill = im_in.copy()
            
            # Mask used to flood filling.
            # Notice the size needs to be 2 pixels than the image.
            h, w = im_in.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            
            # Floodfill from point (0, 0)
            cv2.floodFill(im_floodfill, mask, (0,0), 255);
            
            # Invert floodfilled image
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)
            if len(np.nonzero(im_floodfill_inv)[0]) == 0:
                return im_out
            
            # Combine the two images to get the foreground.
            im_rawout = im_in | im_floodfill_inv
            im_rawout = cv2.dilate(im_rawout, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations = 4)
            im_rawout = cv2.erode(im_rawout, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations = 4)
            im_out = cv2.medianBlur(im_rawout,5)

    return im_out