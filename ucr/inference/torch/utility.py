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
import sys
import argparse
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import math
import torch
import yaml
from pathlib import Path

from ucr.core.architecture import build_architecture
from ucr.core.postprocess import build_postprocess


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--device", type=str, default='cpu')

    # params for text detector
    parser.add_argument("--image_location", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_location", type=str)
    parser.add_argument("--det_config_location", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=1920)
    parser.add_argument("--det_limit_type", type=str, default='min')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    parser.add_argument("--max_batch_size", type=int, default=1)
    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=bool, default=False)

    # CRAFT params
    parser.add_argument(
        '--det_craft_text_thresh', 
        default=0.35, type=float, #BH: 0.5
        help='text confidence threshold') #This threshold is not used in our case.
    parser.add_argument(
        '--det_craft_min_size', 
        default=3, type=float, 
        help='minimum text box size') #0.003 was used with 0.2-tt and 0.1-lt #BH: 0.36
    parser.add_argument(
        '--det_craft_link_thresh', 
        default=0.1, type=float, help='link confidence threshold')
    parser.add_argument(
        '--det_craft_rotated_box', 
        type=str2bool, default=True, 
        help='use this to get rotated rectangles (bounding box)') # Currently not handling for rotated boxes
    parser.add_argument(
        '--det_craft_use_dilate', 
        type=str2bool, default=False, 
        help='use this to specify x_dilation and y_dilation')
    parser.add_argument('--det_craft_xdilate', default=9, type=int, help='left x-padding during post processing')
    parser.add_argument('--det_craft_ydilate', default=3, type=int, help='up y-padding during post processing')
    parser.add_argument('--det_craft_xpad', default=4, type=int, help='x padding of bounding boxes')
    parser.add_argument('--det_craft_ypad', default=2, type=int, help='y padding of bounding boxes')

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_location", type=str)
    parser.add_argument("--rec_config_location", type=str)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch_sim')
    parser.add_argument("--rec_whitelist", type=str, default='') #whitelist characters while prediction
    parser.add_argument("--rec_blacklist", type=str, default='') #blacklist characters while prediction    
    parser.add_argument("--rec_batch_num", type=int, default=30)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./ucr/utils/dict/ench_dict.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="./ucr/utils/fonts/only_en.ttf")
    parser.add_argument("--drop_score", type=float, default=0.)

    # params for text classifier
    parser.add_argument("--cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_location", type=str)
    parser.add_argument("--cls_config_location", type=str)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=8)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    # params for merging resulting values
    parser.add_argument("--merge_boxes", type=str2bool, default=False) # BH: True
    parser.add_argument("--merge_slope_thresh", type=float, default=0.1)
    parser.add_argument("--merge_ycenter_thresh", type=float, default=0.8) # BH: 0.8
    parser.add_argument("--merge_height_thresh", type=float, default=0.6) # BH: 0.6
    parser.add_argument("--merge_width_thresh", type=float, default=2.0) # BH: 2.0
    parser.add_argument("--merge_add_margin", type=float, default=0.05)                          

    return parser.parse_args()


def create_predictor(args, mode, logger):
    if mode == "det":
        model = args.det_model_location
        config = args.det_config_location
    elif mode == 'cls':
        model = args.cls_model_location
        config = args.cls_config_location
    else:
        model = args.rec_model_location
        config = args.rec_config_location

    if model is None:
        logger.info("not find {} model file path {}".format(mode, model))
        sys.exit(0)
    
    if not os.path.exists(model):
        logger.info("not find model file path {}".format(model))
        sys.exit(0)
    if not os.path.exists(config):
        logger.info("not find params file path {}".format(config))
        sys.exit(0)
    config = yaml.load(open(config, 'rb'), Loader=yaml.Loader)
    
    general_params = config['General']
    if mode=='rec':
        general_params["character_dict_path"] = args.rec_char_dict_path
    post_process_class = build_postprocess(config["Postprocess"], general_params)
    
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    
    predictor = build_architecture(config['Architecture'])
    
    device = args.device
    if device == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        logger.info("wrong device selected! Choose eiter 'cuda' or 'cpu'")
        sys.exit(0)
    
    predictor.load_state_dict(torch.load(model, map_location=device))
    predictor.eval()

    return predictor
    

def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./ucr/utils/fonts/only_en.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./ucr/utils/fonts/only_en.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):

        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 200), random.randint(0, 200),
                 random.randint(0, 200))
        draw_left.line(
            [
                (box[0][0], box[0][1]), (box[1][0] - 3, box[1][1]), (box[2][0] - 3,
                box[2][1]), (box[3][0], box[3][1]), (box[0][0], box[0][1]) 
            ],
            fill=color, width=4)
        
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
            draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0] - 3, box[1][1], box[2][0] - 3,
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        else:
            img_fraction = 1.05
            # font_size=1
            font_change=False
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size)
            while font.getsize(txt)[0] > img_fraction*box_width:
                # iterate until the text size is just larger than the criteria
                font_change=True
                font_size -= 1
                font = ImageFont.truetype(font_path, font_size)
            if font_change:
                font_size +=1
            font = ImageFont.truetype(font_path, font_size)
            draw_right.text(
                [box[0][0]+3, box[0][1]+3], txt, fill=(0, 0, 0), font=font)

            wid,het = font.getsize(txt)
            draw_right.polygon(
            [
                box[0][0], box[0][1], box[0][0] + wid + 6, box[0][1], box[0][0] + wid + 6,
                box[0][1] + het + 6, box[0][0], box[0][1] + het + 6
            ],
            outline=color)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def base64_to_cv2(b64str):
    import base64
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image


if __name__ == '__main__':
    test_img = "./doc/test_v2"
    predict_txt = "./doc/predict.txt"
    f = open(predict_txt, 'r')
    data = f.readlines()
    img_path, anno = data[0].strip().split('\t')
    img_name = os.path.basename(img_path)
    img_path = os.path.join(test_img, img_name)
    image = Image.open(img_path)

    data = json.loads(anno)
    boxes, txts, scores = [], [], []
    for dic in data:
        boxes.append(dic['points'])
        txts.append(dic['transcription'])
        scores.append(round(dic['scores'], 3))

    new_img = draw_ocr(image, boxes, txts, scores)

    cv2.imwrite(img_name, new_img)
