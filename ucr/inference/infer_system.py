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

import os
from hydra.experimental.initialize import initialize_config_dir
import tqdm
import cv2
import copy
import numpy as np
import time
from PIL import Image
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize_config_dir

import sys
__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, '../..'))

import ucr.inference.infer_rec as infer_rec
import ucr.inference.infer_det as infer_det
import ucr.inference.infer_cls as infer_cls
from ucr.utils.utility import get_image_file_list, check_and_read_gif
from ucr.utils.logging import get_logger
from ucr.utils.annotation import draw_ocr_box_txt

logger = get_logger()

class TextSystem(object):
    def __init__(self, config):
        self.text_detector = infer_det.TextDetector(config['Detection'])
        self.text_recognizer = infer_rec.TextRecognizer(config['Recognition'])
        self.text_classifier = infer_cls.TextClassifier(config['Classification'])
        
        self.drop_score = config['drop_score']            
        self.merge_boxes = config['merge_boxes']

        if self.merge_boxes:
            self.merge_slope_thresh = config['merge_slope_thresh']
            self.merge_ycenter_thresh = config['merge_ycenter_thresh']
            self.merge_height_thresh = config['merge_height_thresh']
            self.merge_width_thresh = config['merge_width_thresh']
            self.merge_add_margin = config['merge_add_margin']


    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        logger.info("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = self.sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
            
        img_crop_list, angle_list, elapse = self.text_classifier(
            img_crop_list)
        logger.info("cls num  : {}, elapse : {}".format(
            len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        
        if self.drop_score!=0.:
            for box, rec_reuslt in zip(dt_boxes, rec_res):
                text, score = rec_reuslt
                if score >= self.drop_score:
                    filter_boxes.append(box)
                    filter_rec_res.append(rec_reuslt)
        
        else:
            filter_boxes = dt_boxes
            filter_rec_res = rec_res
                
        if self.merge_boxes:
            free_box, free_text, merged_box, merged_text = self.merge_text_boxes(
                filter_boxes, filter_rec_res,
                slope_thresh = self.merge_slope_thresh,
                ycenter_thresh = self.merge_ycenter_thresh,
                height_thresh = self.merge_height_thresh,
                width_thresh = self.merge_width_thresh,
                add_margin = self.merge_add_margin)
            dt_boxes = free_box + merged_box
            rec_res = free_text + merged_text
            
        return dt_boxes, rec_res

    def merge_text_boxes(self,dt_boxes, rec_res, **params):
        dt_boxes = np.asarray(dt_boxes)
        polys = np.empty((len(dt_boxes), 8))
        polys[:,0] = dt_boxes[:,0,0]
        polys[:,1] = dt_boxes[:,0,1]
        polys[:,2] = dt_boxes[:,1,0]
        polys[:,3] = dt_boxes[:,1,1]
        polys[:,4] = dt_boxes[:,2,0]
        polys[:,5] = dt_boxes[:,2,1]
        polys[:,6] = dt_boxes[:,3,0]
        polys[:,7] = dt_boxes[:,3,1]
        slope_ths = params["slope_thresh"]
        ycenter_ths = params["ycenter_thresh"]
        height_ths = params["height_thresh"]
        width_ths = params["width_thresh"]
        add_margin = params["add_margin"]

        horizontal_list, free_list_box, free_list_text, combined_list, merged_list_box, merged_list_text = [],[],[],[],[],[]

        for i, poly in enumerate(polys):
            slope_up = (poly[3]-poly[1])/np.maximum(10, (poly[2]-poly[0]))
            slope_down = (poly[5]-poly[7])/np.maximum(10, (poly[4]-poly[6]))
            if max(abs(slope_up), abs(slope_down)) < slope_ths:
                x_max = max([poly[0],poly[2],poly[4],poly[6]])
                x_min = min([poly[0],poly[2],poly[4],poly[6]])
                y_max = max([poly[1],poly[3],poly[5],poly[7]])
                y_min = min([poly[1],poly[3],poly[5],poly[7]])
                horizontal_list.append([x_min, x_max, y_min, y_max, 0.5*(y_min+y_max), y_max-y_min, rec_res[i][0], rec_res[i][1],str(poly)])
            else:
                height = np.linalg.norm( [poly[6]-poly[0],poly[7]-poly[1]])
                margin = int(1.44*add_margin*height)
                theta13 = abs(np.arctan( (poly[1]-poly[5])/np.maximum(10, (poly[0]-poly[4]))))
                theta24 = abs(np.arctan( (poly[3]-poly[7])/np.maximum(10, (poly[2]-poly[6]))))
                # do I need to clip minimum, maximum value here?
                x1 = poly[0] - np.cos(theta13)*margin
                y1 = poly[1] - np.sin(theta13)*margin
                x2 = poly[2] + np.cos(theta24)*margin
                y2 = poly[3] - np.sin(theta24)*margin
                x3 = poly[4] + np.cos(theta13)*margin
                y3 = poly[5] + np.sin(theta13)*margin
                x4 = poly[6] - np.cos(theta24)*margin
                y4 = poly[7] + np.sin(theta24)*margin

                free_list_box.append(np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]))
                free_list_text.append([rec_res[i][0], rec_res[i][1],str(poly), rec_res[i][0]])

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
                if (abs(np.mean(b_height) - poly[5]) < height_ths*np.mean(b_height)) and (abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths*np.mean(b_height)):
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
            if len(boxes) == 1: # one box per line
                box = boxes[0]
                margin = int(add_margin*min(box[1]-box[0],box[5]))
                _x0 = _x3 = box[0]-margin
                _y0 = _y1 = box[2]-margin
                _x1 = _x2 = box[1]+margin
                _y2 = _y3 = box[3]+margin
                merged_list_box.append(np.array([[_x0,_y0],[_x1,_y1],[_x2,_y2],[_x3,_y3]]))
                merged_list_text.append([box[6], box[7], box[8], box[6]])
            else: # multiple boxes per line
                boxes = sorted(boxes, key=lambda item: item[0])

                merged_box, new_box = [],[]
                for box in boxes:
                    if len(new_box) == 0:
                        b_height = [box[5]]
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        if abs(box[0]-x_max) < width_ths *(box[3]-box[2]): # merge boxes
                            x_max = box[1]
                            new_box.append(box)
                        else:
                            if (abs(np.mean(b_height) - box[5]) < height_ths*np.mean(b_height)) and (abs(box[0]-x_max) < width_ths *(box[3]-box[2])): # merge boxes
                                b_height.append(box[5])
                                x_max = box[1]
                                new_box.append(box)
                            else:
                                b_height = [box[5]]
                                x_max = box[1]
                                merged_box.append(new_box)
                                new_box = [box]
                if len(new_box) >0: merged_box.append(new_box)

                for mbox in merged_box:
                    if len(mbox) != 1: # adjacent box in same line
                        # do I need to add margin here?
                        x_min = min(mbox, key=lambda x: x[0])[0]
                        x_max = max(mbox, key=lambda x: x[1])[1]
                        y_min = min(mbox, key=lambda x: x[2])[2]
                        y_max = max(mbox, key=lambda x: x[3])[3]
                        text_comb = str(mbox[0][6]) if isinstance(mbox[0][6], str) else ''
                        sum_score = mbox[0][7]
                        box_id = str(mbox[0][8])
                        text_id = str(mbox[0][6]) if isinstance(mbox[0][6], str) else ''
                        for val in range(len(mbox)-1):
                            if isinstance(mbox[val+1][6], str):
                                strin = mbox[val+1][6]
                            else:
                                strin = ''
                            text_comb += ' ' + strin
                            sum_score += mbox[val+1][7]
                            box_id += '|||' + str(mbox[val+1][8])
                            text_id += '|||' + strin 
                        avg_score = sum_score / len(mbox)
                        margin = int(add_margin*(y_max - y_min))

                        # merged_list.append([x_min-margin, x_max+margin, y_min-margin, y_max+margin, text_comb, avg_score])
                        _x0 = _x3 = x_min-margin
                        _y0 = _y1 = y_min-margin
                        _x1 = _x2 = x_max+margin
                        _y2 = _y3 = y_max+margin
                        merged_list_box.append(np.array([[_x0,_y0],[_x1,_y1],[_x2,_y2],[_x3,_y3]]))
                        merged_list_text.append([text_comb, avg_score, box_id, text_id])

                    else: # non adjacent box in same line
                        box = mbox[0]

                        margin = int(add_margin*(box[3] - box[2]))
                        # merged_list.append([box[0]-margin,box[1]+margin,box[2]-margin,box[3]+margin, box[6], box[7]])
                        _x0 = _x3 = box[0]-margin
                        _y0 = _y1 = box[2]-margin
                        _x1 = _x2 = box[1]+margin
                        _y2 = _y3 = box[3]+margin
                        merged_list_box.append(np.array([[_x0,_y0],[_x1,_y1],[_x2,_y2],[_x3,_y3]]))
                        merged_list_text.append([box[6], box[7], box[8], box[6]])

        # may need to check if box is really in image
        return free_list_box, free_list_text, merged_list_box, merged_list_text
    
    def sorted_boxes(self, dt_boxes):
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
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

def main():        
    cfg_dir = hydra.utils.to_absolute_path('conf')

    with initialize_config_dir(config_dir=cfg_dir, job_name="infer_det"):
        cfg = compose(config_name="infer_det")    
        print(cfg.pretty())    
        config_det = OmegaConf.to_container(cfg)
        
    with initialize_config_dir(config_dir=cfg_dir, job_name="infer_rec"):
        cfg = compose(config_name="infer_rec")        
        config_rec = OmegaConf.to_container(cfg)
        
    with initialize_config_dir(config_dir=cfg_dir, job_name="infer_cls"):
        cfg = compose(config_name="infer_cls")        
        config_cls = OmegaConf.to_container(cfg)
    
    with initialize_config_dir(config_dir=cfg_dir, job_name="infer_system"):
        cfg = compose(config_name="infer_system")        
        config = OmegaConf.to_container(cfg)
        
    config['Detection'] = config_det
    config['Recognition'] = config_rec
    config['Classification'] = config_cls
    
    input_location = hydra.utils.to_absolute_path(config['input_location'])
    image_file_list = get_image_file_list(input_location)
    
    text_sys = TextSystem(config)
    is_visualize = config['is_visualize']
    font_path = hydra.utils.to_absolute_path(config_rec['font_path'])
    drop_score = config['drop_score']
    
    
    output_location = hydra.utils.to_absolute_path(config['output_location'])

    for image_file in tqdm.tqdm(image_file_list):
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        logger.info("Predict time of %s: %.3fs" % (image_file, elapse))

        if config['merge_boxes']:
            for text, score, _,  _ in rec_res:
                logger.info("{}, {:.3f}".format(text, score))
        else:
            for text, score in rec_res:
                logger.info("{}, {:.3f}".format(text, score))

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            if not os.path.exists(output_location):
                os.makedirs(output_location)
            cv2.imwrite(
                os.path.join(output_location, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            logger.info("The visualized image saved in {}".format(
                os.path.join(output_location, os.path.basename(image_file))))

if __name__ == "__main__":
    main()
