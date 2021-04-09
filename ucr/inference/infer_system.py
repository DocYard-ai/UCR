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

import copy
import os
import sys

import cv2
import hydra
import numpy as np
import pandas as pd
from hydra.experimental import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, "../.."))

import ucr.inference.infer_cls as infer_cls
import ucr.inference.infer_det as infer_det
import ucr.inference.infer_rec as infer_rec
from ucr.utils.annotation import draw_ocr_box_txt, draw_text_det_res
from ucr.utils.utility import (
    check_and_read_gif,
    download_with_progressbar,
    get_image_file_list,
    merge_text_boxes,
    sorted_boxes,
)


class TextSystem(object):
    def __init__(self, config):
        self.text_detector = infer_det.TextDetector(config["Detection"])
        self.text_recognizer = infer_rec.TextRecognizer(config["Recognition"])
        self.text_classifier = infer_cls.TextClassifier(
            config["Classification"]
        )
        self.drop_score = config["drop_score"]
        self.merge_boxes = config["merge_boxes"]
        self.font_path = config["Recognition"]["font_path"]

        if self.merge_boxes:
            self.merge_slope_thresh = config["merge_slope_thresh"]
            self.merge_ycenter_thresh = config["merge_ycenter_thresh"]
            self.merge_height_thresh = config["merge_height_thresh"]
            self.merge_width_thresh = config["merge_width_thresh"]
            self.merge_add_margin = config["merge_add_margin"]

    def get_rotate_crop_image(self, img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def perform_ocr(self, img, cls, verbose=False):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        if verbose:
            print(
                "\n------------------------dt_boxes num : {},\telapse : {:.3f}------------------------".format(
                    len(dt_boxes), elapse
                )
            )
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        if cls:
            img_crop_list, _, elapse = self.text_classifier(img_crop_list)
            if verbose:
                print(
                    "------------------------cls num  : {},\telapse : {:.3f}------------------------".format(
                        len(img_crop_list), elapse
                    )
                )

        rec_res, elapse = self.text_recognizer(img_crop_list)
        if verbose:
            print(
                "------------------------rec_res num  : {},\telapse : {:.3f}------------------------\n".format(
                    len(rec_res), elapse
                )
            )

        filter_boxes, filter_rec_res = [], []

        if self.drop_score != 0.0:
            for box, rec_reuslt in zip(dt_boxes, rec_res):
                _, score = rec_reuslt
                if score >= self.drop_score:
                    filter_boxes.append(box)
                    filter_rec_res.append(rec_reuslt)
        else:
            filter_boxes = dt_boxes
            filter_rec_res = rec_res

        if self.merge_boxes:
            free_box, free_text, merged_box, merged_text = merge_text_boxes(
                filter_boxes,
                filter_rec_res,
                slope_thresh=self.merge_slope_thresh,
                ycenter_thresh=self.merge_ycenter_thresh,
                height_thresh=self.merge_height_thresh,
                width_thresh=self.merge_width_thresh,
                add_margin=self.merge_add_margin,
            )
            dt_boxes = free_box + merged_box
            rec_res = free_text + merged_text

        return dt_boxes, rec_res

    def __call__(
        self,
        input,
        output="output",
        o="output",
        det=True,
        rec=True,
        cls=False,
        return_type="df",
        save_image=False,
        save_tsv=True,
        verbose=False,
    ):
        assert isinstance(input, (np.ndarray, list, str))

        rec_dict = {}
        out_dict = {}

        if output != "output":
            output = output
        elif o != "output":
            output = o  # output takes higher precedence than o

        if save_tsv or save_image:
            print(
                "Saving prediction results in '{}' folder!\n".format(output),
                flush=True,
            )
            if not os.path.exists(output):
                os.makedirs(output)

        is_imgpath = False
        if isinstance(input, str):
            # download net image
            if input.startswith("http") and input.endswith(".jpg"):
                download_with_progressbar(input, "downloaded.jpg")
                input = "downloaded.jpg"

            input = get_image_file_list(input)
            is_imgpath = True

        elif isinstance(input, np.ndarray):
            input = [input]

        else:
            if isinstance(input[0], str):
                is_imgpath = True

        if len(input) == 0:
            print(
                "NO images found in {}. Please check input location!".format(
                    input
                )
            )
            sys.exit(0)

        print(f"Running Prediction on {len(input)} images:", flush=True)
        i = 0
        with tqdm(
            total=len(input),
            colour="green",
            desc="OCR",
            unit="image",
            unit_scale=True,
            position=0,
        ) as pbar:
            for image in input:
                if is_imgpath:
                    key = image
                    img, flag = check_and_read_gif(image)
                    if not flag:
                        img = cv2.imread(image)
                    if img is None:
                        print("ERROR in loading image:{}".format(image))
                        continue
                else:
                    key = str(i) + ".jpg"
                    i += 1
                    if len(image.shape) == 2:
                        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    elif len(image.shape) == 3:
                        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    elif len(image.shape) == 4:  # png file
                        img = cv2.cvtColor(image[:3], cv2.COLOR_RGB2BGR)
                    else:
                        print(
                            "ERROR: Input array has {} channels. Expected '2', '3' or '4'.".format(
                                len(image.shape)
                            )
                        )
                        continue
                if det:
                    if rec:
                        dt_boxes, rec_res = self.perform_ocr(
                            img, cls, verbose=verbose
                        )
                        if self.merge_boxes:
                            rec_res = rec_res[0:1]

                        if return_type == "list":
                            value = [
                                [box.tolist(), res]
                                for box, res in zip(dt_boxes, rec_res)
                            ]
                        elif return_type == "df" or save_tsv:
                            info_list = [
                                [
                                    int(box[0][0]),
                                    int(box[0][1]),
                                    int(box[2][0]),
                                    int(box[2][1]),
                                    res[0],
                                ]
                                for box, res in zip(dt_boxes, rec_res)
                            ]
                            value = pd.DataFrame(
                                info_list,
                                columns=[
                                    "startX",
                                    "startY",
                                    "endX",
                                    "endY",
                                    "Text",
                                ],
                            )
                            if save_tsv:
                                tsv_out = os.path.join(output, "tsv")
                                if not os.path.exists(tsv_out):
                                    os.makedirs(tsv_out)
                                tsv_path = os.path.join(
                                    tsv_out,
                                    "ocr_{}".format(
                                        os.path.splitext(
                                            os.path.basename(key)
                                        )[0]
                                    ),
                                )
                                value.to_csv(
                                    tsv_path + ".tsv", sep="\t", index=False
                                )

                        if verbose:
                            from tabulate import tabulate

                            headers = ["OCR Result", "Score"]
                            if self.merge_boxes:
                                print(
                                    tabulate(
                                        rec_res[0:1],
                                        headers,
                                        tablefmt="fancy_grid",
                                    )
                                )
                            else:
                                print(
                                    tabulate(
                                        rec_res, headers, tablefmt="fancy_grid"
                                    )
                                )

                        if save_image:
                            image = Image.fromarray(
                                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            )
                            boxes = dt_boxes
                            txts = [rec_res[i][0] for i in range(len(rec_res))]
                            scores = [
                                rec_res[i][1] for i in range(len(rec_res))
                            ]

                            draw_img = draw_ocr_box_txt(
                                image,
                                boxes,
                                txts,
                                scores,
                                drop_score=self.drop_score,
                                font_path=self.font_path,
                            )
                            img_out = os.path.join(output, "image")
                            if not os.path.exists(img_out):
                                os.makedirs(img_out)
                            img_path = os.path.join(
                                img_out,
                                "ocr_{}".format(
                                    os.path.splitext(os.path.basename(key))[0]
                                ),
                            )
                            cv2.imwrite(
                                img_path + ".jpg", draw_img[:, :, ::-1]
                            )
                            pbar.set_postfix_str(
                                "Output Path: '{}'".format(img_path)
                            )

                    else:
                        dt_boxes, elapse = self.text_detector(img)
                        if verbose:
                            print(
                                "\n------------------------dt_boxes num : {},\telapse : {:.3f}------------------------".format(
                                    len(dt_boxes), elapse
                                )
                            )
                        if dt_boxes is not None:
                            if return_type == "list":
                                value = [box.tolist() for box in dt_boxes]

                            elif return_type == "df" or save_tsv:
                                info_list = [
                                    [
                                        int(box[0][0]),
                                        int(box[0][1]),
                                        int(box[2][0]),
                                        int(box[2][1]),
                                    ]
                                    for box in dt_boxes
                                ]
                                value = pd.DataFrame(
                                    info_list,
                                    columns=[
                                        "startX",
                                        "startY",
                                        "endX",
                                        "endY",
                                    ],
                                )
                                if save_tsv:
                                    tsv_out = os.path.join(output, "tsv")
                                    if not os.path.exists(tsv_out):
                                        os.makedirs(tsv_out)
                                    tsv_path = os.path.join(
                                        tsv_out,
                                        "ocr_{}".format(
                                            os.path.splitext(
                                                os.path.basename(key)
                                            )[0]
                                        ),
                                    )
                                    value.to_csv(
                                        tsv_path + ".tsv",
                                        sep="\t",
                                        index=False,
                                    )

                            if save_image:
                                src_im = draw_text_det_res(dt_boxes, img)

                                img_out = os.path.join(output, "image")
                                if not os.path.exists(img_out):
                                    os.makedirs(img_out)
                                img_path = os.path.join(
                                    img_out,
                                    "det_{}".format(
                                        os.path.splitext(
                                            os.path.basename(key)
                                        )[0]
                                    ),
                                )
                                cv2.imwrite(img_path + ".jpg", src_im)
                                pbar.set_postfix_str(
                                    "Output Path: '{}'".format(img_path)
                                )

                    out_dict[key] = value
                else:
                    rec_dict[key] = img
                pbar.update(1)

        if rec_dict:
            img = list(rec_dict.values())
            if cls:
                img, cls_res, _ = self.text_classifier(img)
                if not rec:
                    if verbose:
                        cls_list = [
                            [k, v[0], v[1]]
                            for k, v in zip(list(rec_dict.keys()), cls_res)
                        ]
                        print("Classification Result:\n")
                        from tabulate import tabulate

                        headers = [
                            "File Name",
                            "Classification Result",
                            "Score",
                        ]
                        print(
                            tabulate(cls_list, headers, tablefmt="fancy_grid")
                        )

                    if save_image:
                        img_out = os.path.join(output, "image")
                        if not os.path.exists(img_out):
                            os.makedirs(img_out)
                        output_names = [
                            f"{k}_{v[0]}.jpg"
                            for k, v in zip(list(rec_dict.keys()), cls_res)
                        ]
                        for bno in range(len(img)):
                            cv2.imwrite(
                                os.path.join(img_out, output_names[bno]),
                                img[bno],
                            )

                    cls_output = {
                        k: [v] for k, v in zip(list(rec_dict.keys()), cls_res)
                    }
                    return cls_output
            if rec:
                rec_res, _ = self.text_recognizer(img)
                if verbose:
                    rec_list = [
                        [k, v[0], v[1]]
                        for k, v in zip(list(rec_dict.keys()), rec_res)
                    ]
                    print("Recognition Result:\n")
                    from tabulate import tabulate

                    headers = ["File Name", "Recognition Result", "Score"]
                    print(tabulate(rec_list, headers, tablefmt="fancy_grid"))

                if save_image:
                    img_out = os.path.join(output, "image")
                    if not os.path.exists(img_out):
                        os.makedirs(img_out)
                    output_names = [
                        f"{k}_{v[0]}.jpg"
                        for k, v in zip(list(rec_dict.keys()), rec_res)
                    ]
                    for bno in range(len(img)):
                        cv2.imwrite(
                            os.path.join(img_out, output_names[bno]), img[bno]
                        )

                rec_output = {
                    k: [v] for k, v in zip(list(rec_dict.keys()), rec_res)
                }
                return rec_output
        else:
            return out_dict


def main(args):
    cfg_dir = hydra.utils.to_absolute_path(args.config_dir)

    with initialize_config_dir(
        config_dir=cfg_dir, job_name=args.config_det_name
    ):
        cfg = compose(config_name=args.config_det_name)
        # print("Detection config:\n{}\n".format(cfg.pretty()))
        config_det = OmegaConf.to_container(cfg)

    with initialize_config_dir(
        config_dir=cfg_dir, job_name=args.config_rec_name
    ):
        cfg = compose(config_name=args.config_rec_name)
        # print("Recognition config:\n{}\n".format(cfg.pretty()))
        config_rec = OmegaConf.to_container(cfg)

    with initialize_config_dir(
        config_dir=cfg_dir, job_name=args.config_cls_name
    ):
        cfg = compose(config_name=args.config_cls_name)
        # print("Classification config:\n{}\n".format(cfg.pretty()))
        config_cls = OmegaConf.to_container(cfg)

    with initialize_config_dir(
        config_dir=cfg_dir, job_name=args.config_system_name
    ):
        cfg = compose(config_name=args.config_system_name)
        # print("Final config:\n{}\n".format(cfg.pretty()))
        config = OmegaConf.to_container(cfg)

    config["Detection"] = config_det
    config["Recognition"] = config_rec
    config["Classification"] = config_cls

    text_sys = TextSystem(config)

    input = hydra.utils.to_absolute_path(config["input"])
    _ = text_sys(
        input,
        output=hydra.utils.to_absolute_path(config["output"]),
        det=config["det"],
        rec=config["rec"],
        cls=config["cls"],
        return_type=config["return_type"],
        save_image=config["save_image"],
        save_tsv=config["save_tsv"],
        verbose=config["verbose"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="ucr/conf")
    parser.add_argument("--config_det_name", type=str, default="infer_det")
    parser.add_argument("--config_rec_name", type=str, default="infer_rec")
    parser.add_argument("--config_cls_name", type=str, default="infer_cls")
    parser.add_argument(
        "--config_system_name", type=str, default="infer_system"
    )

    args = parser.parse_args()
    main(args)
