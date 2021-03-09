# Copyright (c) 2020 FSMLP Authors. All Rights Reserved.

import numpy as np
import cv2


class CRAFTPreprocessTest(object):
    
    def __init__(self, params):
        self.canvas_size = params['canvas_size']
        self.mag_ratio = params['mag_ratio']
        self.interpolation = params['interpolation']


    def resize_image(self, im):

        # im = cv2.GaussianBlur(im,(6,6),0)

        kernel = np.ones((2,1), np.uint8) 


        height, width, channel = im.shape
        im = cv2.dilate(im, kernel, iterations=1) 

        # magnify image size
        target_size = self.mag_ratio * max(height, width)

        # set original image size
        if target_size > self.canvas_size:
            target_size = self.canvas_size
        
        ratio = target_size / max(height, width)    

        target_h, target_w = int(height * ratio), int(width * ratio)
        proc = cv2.resize(im, (target_w, target_h), interpolation = self.interpolation)


        # make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = proc
        target_h, target_w = target_h32, target_w32

        size_heatmap = (int(target_w/2), int(target_h/2))

        return resized, ratio, size_heatmap


    def normalize(self, im):
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im[:, :, 0] -= img_mean[0]
        im[:, :, 1] -= img_mean[1]
        im[:, :, 2] -= img_mean[2]
        im[:, :, 0] /= img_std[0]
        im[:, :, 1] /= img_std[1]
        im[:, :, 2] /= img_std[2]
        return im
        

    def __call__(self, im):
        im, ratio, size_heatmap = self.resize_image(im)
        im = self.normalize(im)
        im = im.transpose(2, 0, 1)
        im = im[np.newaxis, :]
        return im, ratio, size_heatmap