# Copyright (c) 2020 FSMLP Authors. All Rights Reserved.
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import cv2
import math
class CRAFTPostProcess(object):

    def __init__(self, 
                 text_thresh=0.35,
                 link_thresh=0.1,
                 min_size=3,
                 use_dilate=False,
                 xpad=4,
                 ypad=2,
                 xdilate=9,
                 ydilate=3,
                 merged_dilation_kernel=None,
                 rotated_box=True, **kwargs):
        self.max_candidates = 1000
        self.text_thresh = text_thresh
        self.link_thresh = link_thresh
        self.min_size = min_size
        self.use_dilate = use_dilate
        self.xdilate = xdilate
        self.ydilate = ydilate
        self.xpad = xpad
        self.ypad = ypad
        self.merged_dilation_kernel = None if merged_dilation_kernel is None else np.array(
            [[1, 1], [1, 1]])
        self.rotated_box = rotated_box

    def boxes_from_bitmap0(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''
        
        bitmap = _bitmap
        height, width = bitmap.shape
        contours = None

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))

            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
            
        return np.array(boxes, dtype=np.int16), scores
    
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''
        
        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        xpad = self.xpad
        ypad = self.ypad
        
        padding = np.array([[-xpad, -ypad], [+xpad, -ypad], [+xpad, +ypad], [-xpad, +ypad]])
        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            box = np.array(points)
            
            boxes.append(box)
        
        if len(boxes)==0:
            return [],[]
        
        
        boxes = np.array(boxes)
        
        boxes = np.add(boxes, padding)
        boxes[boxes[:,:,0]<0] = 0
        boxes[boxes[:,1,0]>width] = width 
        boxes[boxes[:,2,0]>width] = width
        boxes[boxes[:,3,1]>height] = height 
        boxes[boxes[:,3,1]>height] = height
        
        boxes[:,:, 0] = np.clip(
            np.round(boxes[:,:, 0] / width * dest_width), 0, dest_width)
        boxes[:,:, 1] = np.clip(
            np.round(boxes[:,:, 1] / height * dest_height), 0, dest_height)
        return boxes, scores

    def boxes_from_bitmap1(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''
        
        bitmap = _bitmap
        height, width = bitmap.shape

        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bitmap.astype(np.uint8), connectivity=4)

        boxes = []
        scores = []
        for k in range(1,nLabels):
            # size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10: continue

            # make segmentation map
            segmap = np.zeros(bitmap.shape, dtype=np.uint8)
            segmap[labels==k] = 255 # remove link area
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            # boundary check
            if sx < 0 : sx = 0
            if sy < 0 : sy = 0
            if ex >= width: ex = width
            if ey >= height: ey = height
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # make box
            contour = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
            rectangle = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rectangle)

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(contour[:,0]), max(contour[:,0])
                t, b = min(contour[:,1]), max(contour[:,1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4-startidx, 0)
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)

            boxes.append(box)

        return boxes, scores

    def unclip(self, box):
        from shapely.geometry import Polygon        
        import pyclipper
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        
        if self.rotated_box:
            bounding_box = cv2.minAreaRect(contour)
            points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

            index_1, index_2, index_3, index_4 = 0, 1, 2, 3
            if points[1][1] > points[0][1]:
                index_1 = 0
                index_4 = 1
            else:
                index_1 = 1
                index_4 = 0
            if points[3][1] > points[2][1]:
                index_2 = 2
                index_3 = 3
            else:
                index_2 = 3
                index_3 = 2

            box = [
                points[index_1], points[index_2], points[index_3], points[index_4]
            ]
            
            return box, min(bounding_box[1])
        
        else:
            l, t, w, h = cv2.boundingRect(contour)
            box = [[l, t], [l+w, t], [l+w, t+h], [l, t+h]]
            
            return box, min(w,h)
        

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):

        textmap = outs_dict['text_map']
        linkmap = outs_dict['link_map']
        
        if torch.is_tensor(textmap):
            textmap = textmap.numpy()
        
        if torch.is_tensor(linkmap):
            linkmap = linkmap.numpy()
            
        text_threshold = self.text_thresh
        link_threshold = self.link_thresh

        use_dilate = self.use_dilate
        xdilate = self.xdilate
        ydilate = self.ydilate

        boxes_batch = []
        for batch_index in range(textmap.shape[0]):
            # perform thresholding on greyscale textmap
            _, text_score = cv2.threshold(textmap[batch_index], text_threshold, 1, 0)

            if use_dilate:
                # custom kernel defined to pad the textmap strictly in the upper left region of each blob
                center_x = int((xdilate - 1) / 2)
                center_y = int((ydilate - 1) / 2)

                inner = np.ones(center_x * center_y).reshape(center_y, center_x).astype(np.uint8)
                outer_r = np.zeros((xdilate - center_x) * center_y).reshape(center_y, (xdilate - center_x)).astype(np.uint8)
                outer_d = np.zeros((xdilate) * -1 * (center_y - ydilate)).reshape(ydilate - center_y, xdilate).astype(np.uint8)

                final = np.append(outer_r, inner, 1)
                Vkernel = np.append(outer_d, final, 0)

                # dilation is performed here
                text_score = cv2.dilate(text_score, Vkernel, 1)

            # perform thresholding on greyscale linkmap
            _, link_score = cv2.threshold(linkmap[batch_index], link_threshold, 1, 0)
            text_score_comb = np.clip(text_score + link_score, 0, 1)
            
            src_h, src_w, _, _ = shape_list[batch_index]
            if self.merged_dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(text_score_comb).astype(np.uint8),
                    self.merged_dilation_kernel)
            else:
                mask = text_score_comb
                
            boxes, _ = self.boxes_from_bitmap(textmap[batch_index], mask,
                                                   src_w, src_h)

            boxes_batch.append({'points': boxes})
            
        return boxes_batch