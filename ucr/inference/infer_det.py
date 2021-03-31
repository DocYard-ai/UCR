# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Modifications copyright (c) 2021 DocYard Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import time

import hydra
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

import cv2
import torch

import ucr.utils.annotation as utility
from ucr.core.architecture import build_architecture
from ucr.core.postprocess import build_postprocess
from ucr.core.preprocess import build_preprocess, preprocess
from ucr.utils.utility import check_and_read_gif, get_image_file_list


class TextDetector(object):
    def __init__(self, config):
        self.device = config["device"]
        self.det_algorithm = config["Architecture"]["algorithm"]
        self.preprocess_op = build_preprocess(config["Preprocess"])
        self.postprocess_op = build_postprocess(config["Postprocess"])
        self.predictor = build_architecture(config["Architecture"])

        if self.device == "cuda":
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        elif self.device == "cpu":
            self.device = torch.device("cpu")
        else:
            log.error("wrong device selected! Choose eiter 'cuda' or 'cpu'")
            sys.exit(0)

        self.predictor.load_state_dict(
            torch.load(config["model_location"], map_location=self.device)
        )
        self.predictor.eval()

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        starttime = time.time()
        data = {"image": img}
        data = preprocess(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)

        img = torch.as_tensor(img)
        input_tensors = img.to(self.device)
        self.predictor.to(self.device)

        output_tensors = self.predictor(input_tensors)
        outputs = []
        for output_tensor in output_tensors.values():
            output = output_tensor.cpu().data.numpy()
            outputs.append(output)

        preds = {}
        if self.det_algorithm == "CRAFT":
            preds["text_map"] = outputs[0]
            preds["link_map"] = outputs[1]

        elif self.det_algorithm == "EAST":
            preds["f_geo"] = outputs[0]
            preds["f_score"] = outputs[1]
        elif self.det_algorithm == "SAST0" or self.det_algorithm == "SAST1":
            preds["f_border"] = outputs[0]
            preds["f_score"] = outputs[1]
            preds["f_tco"] = outputs[2]
            preds["f_tvo"] = outputs[3]
        elif self.det_algorithm == "DB":
            preds["maps"] = outputs[0]
        else:
            raise NotImplementedError

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]
        if (
            self.det_algorithm == "SAST0"
        ):  # TODO: Implement two versions of SAST both for det_sast_polygon True and False state
            dt_boxes = self.filter_tag_det_res_only_clip(
                dt_boxes, ori_im.shape
            )
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        elapse = time.time() - starttime
        return dt_boxes, elapse


def main(cfg):
    log.debug("Detection config:\n{}\n".format(cfg.pretty()))
    config = OmegaConf.to_container(cfg)
    GlobalHydra.instance().clear()

    input = hydra.utils.to_absolute_path(config["input"])
    image_file_list = get_image_file_list(input)
    model_location = hydra.utils.to_absolute_path(config["model_location"])
    config["model_location"] = model_location
    text_detector = TextDetector(config)
    total_time = 0
    count = 0
    output = hydra.utils.to_absolute_path(config["output"])
    if not os.path.exists(output):
        os.makedirs(output)
    for image_file in tqdm(
        image_file_list, colour="green", desc="Detection", unit="image"
    ):
        count += 1
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            log.warning("error in loading image:{}".format(image_file))
            continue
        dt_boxes, elapse = text_detector(img)
        total_time += elapse
        src_im = utility.draw_text_det_res(dt_boxes, img)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(output, "det_{}".format(img_name_pure))
        cv2.imwrite(img_path, src_im)
        log.info(
            "[{}/{}] Detection output is saved in --------- {}".format(
                count, len(image_file_list), img_path
            )
        )
    log.info(
        "\nTotal Prediction time for {} images:\t{:.5f} s".format(
            len(image_file_list), total_time
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="../conf")
    parser.add_argument("--config_name", type=str, default="infer_det")

    args = parser.parse_args()

    main_wrapper = hydra.main(
        config_path=args.config_path, config_name=args.config_name
    )
    main_wrapper(main)()
