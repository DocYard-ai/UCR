# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Modifications copyright (c) 2021 DocYard Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import imghdr
import logging
import os
import shutil
import sys

import cv2
import numpy as np
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


def print_dict(d, logger, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            log.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            log.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            log.info("{}{} : {}".format(delimiter * " ", k, v))


def get_check_global_params(mode):
    check_params = [
        "device",
        "max_text_length",
        "image_shape",
        "image_shape",
        "character_type",
        "loss_type",
    ]
    if mode == "train_eval":
        check_params = check_params + [
            "train_batch_size_per_card",
            "test_batch_size_per_card",
        ]
    elif mode == "test":
        check_params = check_params + ["test_batch_size_per_card"]
    return check_params


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "GIF"}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists


def check_and_read_gif(img_path):
    if os.path.basename(img_path)[-3:] in ["gif", "GIF"]:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            # logger = logging.getLogger('ucr')
            log.warning("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True
    return None, False


def download_with_progressbar(url, save_path):
    print("Downloading '{}' to '{}'".format(url, save_path))
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(save_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes:
        print("Something went wrong while downloading model.")
        sys.exit(0)


def maybe_download(model_storage_directory, url, force_download=False):
    # using custom model
    fname = url.split("/")[-1][0:-4]
    tmp_path = os.path.join(model_storage_directory, fname)
    if force_download or not os.path.exists(tmp_path):
        os.makedirs(tmp_path, exist_ok=True)
        download_with_progressbar(url, tmp_path + ".zip")
        shutil.unpack_archive(tmp_path + ".zip", tmp_path, "zip")
        os.remove(tmp_path + ".zip")
    return tmp_path


def merge_text_boxes(dt_boxes, rec_res, **params):
    dt_boxes = np.asarray(dt_boxes)
    polys = np.empty((len(dt_boxes), 8))
    polys[:, 0] = dt_boxes[:, 0, 0]
    polys[:, 1] = dt_boxes[:, 0, 1]
    polys[:, 2] = dt_boxes[:, 1, 0]
    polys[:, 3] = dt_boxes[:, 1, 1]
    polys[:, 4] = dt_boxes[:, 2, 0]
    polys[:, 5] = dt_boxes[:, 2, 1]
    polys[:, 6] = dt_boxes[:, 3, 0]
    polys[:, 7] = dt_boxes[:, 3, 1]
    slope_ths = params["slope_thresh"]
    ycenter_ths = params["ycenter_thresh"]
    height_ths = params["height_thresh"]
    width_ths = params["width_thresh"]
    add_margin = params["add_margin"]

    (
        horizontal_list,
        free_list_box,
        free_list_text,
        combined_list,
        merged_list_box,
        merged_list_text,
    ) = ([], [], [], [], [], [])

    for i, poly in enumerate(polys):
        slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
        slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max([poly[0], poly[2], poly[4], poly[6]])
            x_min = min([poly[0], poly[2], poly[4], poly[6]])
            y_max = max([poly[1], poly[3], poly[5], poly[7]])
            y_min = min([poly[1], poly[3], poly[5], poly[7]])
            horizontal_list.append(
                [
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    0.5 * (y_min + y_max),
                    y_max - y_min,
                    rec_res[i][0],
                    rec_res[i][1],
                    str(poly),
                ]
            )
        else:
            height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
            margin = int(1.44 * add_margin * height)
            theta13 = abs(
                np.arctan(
                    (poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4]))
                )
            )
            theta24 = abs(
                np.arctan(
                    (poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6]))
                )
            )
            # do I need to clip minimum, maximum value here?
            x1 = poly[0] - np.cos(theta13) * margin
            y1 = poly[1] - np.sin(theta13) * margin
            x2 = poly[2] + np.cos(theta24) * margin
            y2 = poly[3] - np.sin(theta24) * margin
            x3 = poly[4] + np.cos(theta13) * margin
            y3 = poly[5] + np.sin(theta13) * margin
            x4 = poly[6] - np.cos(theta24) * margin
            y4 = poly[7] + np.sin(theta24) * margin

            free_list_box.append(
                np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            )
            free_list_text.append(
                [rec_res[i][0], rec_res[i][1], str(poly), rec_res[i][0]]
            )

    horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # combine box
    new_box = []
    for poly in horizontal_list:

        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            # comparable height and comparable y_center level up to ths*height
            if (
                abs(np.mean(b_height) - poly[5])
                < height_ths * np.mean(b_height)
            ) and (
                abs(np.mean(b_ycenter) - poly[4])
                < ycenter_ths * np.mean(b_height)
            ):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # merge list use sort again
    for boxes in combined_list:
        if len(boxes) == 1:  # one box per line
            box = boxes[0]
            margin = int(add_margin * min(box[1] - box[0], box[5]))
            _x0 = _x3 = box[0] - margin
            _y0 = _y1 = box[2] - margin
            _x1 = _x2 = box[1] + margin
            _y2 = _y3 = box[3] + margin
            merged_list_box.append(
                np.array([[_x0, _y0], [_x1, _y1], [_x2, _y2], [_x3, _y3]])
            )
            merged_list_text.append([box[6], box[7], box[8], box[6]])
        else:  # multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [], []
            for box in boxes:
                if len(new_box) == 0:
                    b_height = [box[5]]
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if abs(box[0] - x_max) < width_ths * (
                        box[3] - box[2]
                    ):  # merge boxes
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        if (
                            abs(np.mean(b_height) - box[5])
                            < height_ths * np.mean(b_height)
                        ) and (
                            abs(box[0] - x_max) < width_ths * (box[3] - box[2])
                        ):  # merge boxes
                            b_height.append(box[5])
                            x_max = box[1]
                            new_box.append(box)
                        else:
                            b_height = [box[5]]
                            x_max = box[1]
                            merged_box.append(new_box)
                            new_box = [box]
            if len(new_box) > 0:
                merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1:  # adjacent box in same line
                    # do I need to add margin here?
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]
                    text_comb = (
                        str(mbox[0][6]) if isinstance(mbox[0][6], str) else ""
                    )
                    sum_score = mbox[0][7]
                    box_id = str(mbox[0][8])
                    text_id = (
                        str(mbox[0][6]) if isinstance(mbox[0][6], str) else ""
                    )
                    for val in range(len(mbox) - 1):
                        if isinstance(mbox[val + 1][6], str):
                            strin = mbox[val + 1][6]
                        else:
                            strin = ""
                        text_comb += " " + strin
                        sum_score += mbox[val + 1][7]
                        box_id += "|||" + str(mbox[val + 1][8])
                        text_id += "|||" + strin
                    avg_score = sum_score / len(mbox)
                    margin = int(add_margin * (y_max - y_min))

                    # merged_list.append([x_min-margin, x_max+margin, y_min-margin, y_max+margin, text_comb, avg_score])
                    _x0 = _x3 = x_min - margin
                    _y0 = _y1 = y_min - margin
                    _x1 = _x2 = x_max + margin
                    _y2 = _y3 = y_max + margin
                    merged_list_box.append(
                        np.array(
                            [[_x0, _y0], [_x1, _y1], [_x2, _y2], [_x3, _y3]]
                        )
                    )
                    merged_list_text.append(
                        [text_comb, avg_score, box_id, text_id]
                    )

                else:  # non adjacent box in same line
                    box = mbox[0]

                    margin = int(add_margin * (box[3] - box[2]))
                    # merged_list.append([box[0]-margin,box[1]+margin,box[2]-margin,box[3]+margin, box[6], box[7]])
                    _x0 = _x3 = box[0] - margin
                    _y0 = _y1 = box[2] - margin
                    _x1 = _x2 = box[1] + margin
                    _y2 = _y3 = box[3] + margin
                    merged_list_box.append(
                        np.array(
                            [[_x0, _y0], [_x1, _y1], [_x2, _y2], [_x3, _y3]]
                        )
                    )
                    merged_list_text.append([box[6], box[7], box[8], box[6]])

    # may need to check if box is really in image
    return free_list_box, free_list_text, merged_list_box, merged_list_text


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    config:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and (
            _boxes[i + 1][0][0] < _boxes[i][0][0]
        ):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes
