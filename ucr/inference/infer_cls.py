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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hydra
from omegaconf import OmegaConf
import numpy as np
import traceback
import os
import sys


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import torch
import cv2
import copy
import time

from ucr.core.preprocess import build_preprocess, preprocess
from ucr.core.postprocess import build_postprocess
from ucr.core.architecture import build_architecture
from ucr.utils.utility import get_image_file_list, check_and_read_gif

from ucr.utils.logging import get_logger
logger = get_logger()

class TextClassifier(object):
    def __init__(self, config):
        self.device = config['device']
        
        for op in config['Preprocess']:
            op_name = list(op)[0]
            if op_name == 'ClsResizeImg':
                image_shape = op[op_name]['image_shape'] # TODO:add try except here
                
        self.cls_image_shape = image_shape
        
        self.batch_size = config['batch_size']
        # * Note: implement cls_image_shape in config['Postprocess'] Phase 2
        self.threshold = config['threshold']
        
        label_list = {'label_list': config['label_list']}
        
        self.preprocess_op = build_preprocess(config['Preprocess'])
        self.postprocess_op = build_postprocess(config['Postprocess'], label_list)
        self.predictor = build_architecture(config['Architecture'])

        if self.device == 'cuda':
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif self.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            logger.info("wrong device selected! Choose eiter 'cuda' or 'cpu'")
            sys.exit(0)
        self.predictor.load_state_dict(torch.load(hydra.utils.to_absolute_path(config['model_location']), map_location=self.device))
        self.predictor.eval()

    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [['', 0.0]] * img_num
        batch_num = self.batch_size
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                image = {'image': img_list[indices[ino]]}
                norm_img = preprocess(image, self.preprocess_op)['image']
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            starttime = time.time()

            # self.input_tensor.copy_from_cpu(norm_img_batch)
            input_tensors = torch.as_tensor(norm_img_batch)
            input_tensors = input_tensors.to(self.device)            
            self.predictor.to(self.device)
            output_tensors = self.predictor(input_tensors)
            prob_out = output_tensors.cpu().data.numpy()
            
            cls_result = self.postprocess_op(prob_out)
            elapse += time.time() - starttime
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if '180' in label and score > self.threshold:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1)
        return img_list, cls_res, elapse


@hydra.main(config_path="../../conf/infer_cls.yaml")
def main(cfg):
    config = OmegaConf.to_container(cfg)
    
    input_location = hydra.utils.to_absolute_path(config['input_location'])
    image_file_list = get_image_file_list(input_location)
    text_classifier = TextClassifier(config)
    valid_image_file_list = []
    img_list = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        img_list, cls_res, predict_time = text_classifier(img_list)
    except:
        logger.info(traceback.format_exc())
        logger.info(
            "ERROR!!!! \n"
            "Please read the FAQ：https://github.com/PaddlePaddle/PaddleOCR#faq \n"
            "If your model has tps module:  "
            "TPS does not support variable shape.\n"
            "Please set --rec_image_shape='3,32,100' and --rec_char_type='en' ")
        exit()
    for ino in range(len(img_list)):
        logger.info("Predicts of {}:{}".format(valid_image_file_list[ino],
                                               cls_res[ino]))
    logger.info("Total predict time for {} images, cost: {:.3f}".format(
        len(img_list), predict_time))


if __name__ == "__main__":
    main()

