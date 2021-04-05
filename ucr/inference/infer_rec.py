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

from __future__ import absolute_import, division, print_function

import logging
import os
import sys

import hydra
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

log = logging.getLogger(__name__)
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

import time
import traceback

import cv2
import torch
from tqdm import tqdm

from ucr.core.architecture import build_architecture
from ucr.core.postprocess import build_postprocess
from ucr.core.preprocess import build_preprocess, preprocess
from ucr.core.preprocess.label_ops import BaseRecLabelEncode
from ucr.utils.utility import check_and_read_gif, get_image_file_list


class TextRecognizer(object):
    def __init__(self, config):
        self.device = config["device"]

        for op in config["Preprocess"]:
            op_name = list(op)[0]
            if op_name == "RecResizeImg":
                image_shape = op[op_name][
                    "image_shape"
                ]  # TODO:add try except here

        self.rec_image_shape = image_shape
        self.lang = config["lang"]
        self.rec_batch_num = config["batch_size"]
        self.rec_algorithm = config["Architecture"]["algorithm"]
        self.rec_whitelist = config["whitelist"]
        self.rec_blacklist = config["blacklist"]

        # Todo: Implement whitelist and blacklist in postprocessing step.

        char_dict_location = config["char_dict_location"]
        self.char_ops = BaseRecLabelEncode(
            config["max_text_length"],
            char_dict_location,
            config["lang"],
            config["use_space_char"],
        )

        global_keys = ["lang", "use_space_char", "max_text_length"]
        global_cfg = {
            key: value for key, value in config.items() if key in global_keys
        }
        global_cfg["char_dict_location"] = char_dict_location

        self.preprocess_op = build_preprocess(config["Preprocess"])
        self.postprocess_op = build_postprocess(
            config["Postprocess"], global_cfg
        )

        # build model
        if hasattr(self.postprocess_op, "character"):
            config["Architecture"]["Head"]["out_channels"] = len(
                getattr(self.postprocess_op, "character")
            )
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

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):

                image = {"image": img_list[indices[ino]]}
                norm_img = preprocess(image, self.preprocess_op)

                if self.rec_algorithm != "SRN":
                    norm_img = norm_img["image"][np.newaxis, :]
                    norm_img_batch.append(norm_img)
                else:
                    encoder_word_pos_list = []
                    gsrm_word_pos_list = []
                    gsrm_slf_attn_bias1_list = []
                    gsrm_slf_attn_bias2_list = []
                    encoder_word_pos_list.append(norm_img[1])
                    gsrm_word_pos_list.append(norm_img[2])
                    gsrm_slf_attn_bias1_list.append(norm_img[3])
                    gsrm_slf_attn_bias2_list.append(norm_img[4])
                    norm_img_batch.append(norm_img[0])
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            if self.rec_algorithm == "SRN":
                starttime = time.time()
                encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
                gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
                gsrm_slf_attn_bias1_list = np.concatenate(
                    gsrm_slf_attn_bias1_list
                )
                gsrm_slf_attn_bias2_list = np.concatenate(
                    gsrm_slf_attn_bias2_list
                )

                inputs = [
                    torch.as_tensor(norm_img_batch).to(self.device),
                    torch.as_tensor(encoder_word_pos_list).to(self.device),
                    torch.as_tensor(gsrm_word_pos_list).to(self.device),
                    torch.as_tensor(gsrm_slf_attn_bias1_list).to(self.device),
                    torch.as_tensor(gsrm_slf_attn_bias2_list).to(self.device),
                ]
                self.predictor.to(self.device)

                output_tensors = self.predictor(inputs[0], inputs[1:])
                outputs = []
                for output_tensor in output_tensors.values():
                    output = output_tensor.cpu().data.numpy()
                    outputs.append(output)
                preds = {"predict": outputs[3]}
            else:
                starttime = time.time()
                imgs = torch.as_tensor(norm_img_batch)
                input_tensors = imgs.to(self.device)
                self.predictor.to(self.device)

                output_tensors = self.predictor(input_tensors)

                preds = output_tensors.cpu().data.numpy()
                # If both blacklist and whitelist are provided, whitelist is only used
                if not self.rec_whitelist and self.rec_blacklist:
                    self.mod_chars = np.arange(preds.shape[-1])
                    black_list = self.char_ops.encode(self.rec_blacklist)
                    black_list = np.array(black_list) + 1
                    self.mod_chars = np.setdiff1d(self.mod_chars, black_list)
                elif self.rec_whitelist:
                    white_list = self.char_ops.encode(self.rec_whitelist)
                    self.mod_chars = np.append(white_list, [-1]) + 1
                elif not self.rec_whitelist and not self.rec_blacklist:
                    self.mod_chars = []

                if len(self.mod_chars) != 0:
                    mod_onehot = np.zeros(preds.shape[-1])
                    mod_onehot[self.mod_chars] = 1
                    preds = np.multiply(
                        preds, mod_onehot
                    )  # * Implemented blacklist and whitelist here!

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            elapse += time.time() - starttime
        return rec_res, elapse


def main(cfg):
    log.debug("Recognition config:\n{}\n".format(cfg.pretty()))
    config = OmegaConf.to_container(cfg)
    GlobalHydra.instance().clear()

    input = hydra.utils.to_absolute_path(config["input"])
    model_location = hydra.utils.to_absolute_path(config["model_location"])
    config["model_location"] = model_location
    char_dict_location = hydra.utils.to_absolute_path(
        config["char_dict_location"]
    )
    config["char_dict_location"] = char_dict_location

    image_file_list = get_image_file_list(input)
    text_recognizer = TextRecognizer(config)
    valid_image_file_list = []
    img_list = []
    for image_file in tqdm(
        image_file_list, colour="green", desc="Recognition", unit="image"
    ):
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            log.warning("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        rec_res, predict_time = text_recognizer(img_list)
    except Exception:
        log.error(traceback.format_exc())
        log.error(
            "ERROR!!!! \n"
            "Please read the FAQ：https://github.com/PaddlePaddle/PaddleOCR#faq \n"
            "If your model has tps module:  "
            "TPS does not support variable shape.\n"
            "Please set --rec_image_shape='3,32,100' and --rec_char_type='en' "
        )
        exit()
    for ino in range(len(img_list)):
        log.info(
            "[{}/{}] Prediction ['RESULT', SCORE] of {}:\t{}".format(
                ino + 1,
                len(img_list),
                valid_image_file_list[ino],
                rec_res[ino],
            )
        )
    log.info(
        "\nTotal Prediction time for {} images:\t{:.5f} s".format(
            len(img_list), predict_time
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="../conf")
    parser.add_argument("--config_name", type=str, default="infer_rec")

    args = parser.parse_args()

    main_wrapper = hydra.main(
        config_path=args.config_path, config_name=args.config_name
    )
    main_wrapper(main)()
