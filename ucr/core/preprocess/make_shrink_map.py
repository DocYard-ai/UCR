# -*- coding:utf-8 -*-
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

__all__ = ["MakeShrinkMap"]


class MakeShrinkMap(object):
    r"""
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    """

    def __init__(self, min_text_size=8, shrink_ratio=0.4, **kwargs):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, data):
        image = data["image"]
        text_polys = data["polys"]
        ignore_tags = data["ignore_tags"]

        h, w = image.shape[:2]
        text_polys, ignore_tags = self.validate_polygons(
            text_polys, ignore_tags, h, w
        )
        gt = np.zeros((h, w), dtype=np.float32)
        # gt = np.zeros((1, h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(
                    mask, polygon.astype(np.int32)[np.newaxis, :, :], 0
                )
                ignore_tags[i] = True
            else:
                polygon_shape = Polygon(polygon)
                distance = (
                    polygon_shape.area
                    * (1 - np.power(self.shrink_ratio, 2))
                    / polygon_shape.length
                )
                subject = [tuple(text) for text in text_polys[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(
                    subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON
                )
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(
                        mask, polygon.astype(np.int32)[np.newaxis, :, :], 0
                    )
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                # cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)

        data["shrink_map"] = gt
        data["shrink_mask"] = mask
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        """
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        """
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        # return cv2.contourArea(polygon.astype(np.float32))
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (
                polygon[next_index, 1] - polygon[i, 1]
            )

        return edge / 2.0
