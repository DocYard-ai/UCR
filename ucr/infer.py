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

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import tarfile

from ucr.inference.torch import predict_system
from ucr.utils.logging import get_logger

logger = get_logger()
from ucr.utils.utility import check_and_read_gif, get_image_file_list

__all__ = ['UCR']

model_urls = {
    'torch_server': {
        'det': {
            'DB': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_server/det_ench_ppocr_server.tar'
            },
            'CRAFT': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_server/det_vgg_craft.tar'
            }    
        },
        'rec': {
            'ch_sim': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_server/rec_ench_ppocr_server.tar',
                'dict_path': 'utils/dict/ench_dict.txt',
                
            }
        }
    },

    'torch_mobile': {
        'det': {
            'DB': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/det_ench_ppocr_mobile.tar',
            }  
        },
        'rec': {
            'ch_sim': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_ench_ppocr_mobile.tar',
                'dict_path': 'utils/dict/ench_dict.txt',
                
            },
            'en': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_en_number_mobile.tar',
                'dict_path': 'utils/dict/en_dict.txt',
                
            },
            'fr': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_french_mobile.tar',
                'dict_path': 'utils/dict/french_dict.txt',
                
            },
            'de': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_german_mobile.tar',
                'dict_path': 'utils/dict/german_dict.txt',
                
            },
            'ko': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_korean_mobile.tar',
                'dict_path': 'utils/dict/korean_dict.txt',
                
            },
            'ja': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_japan_mobile.tar',
                'dict_path': 'utils/dict/japan_dict.txt',
                
            },
            'ch_tra': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_chinese_cht_mobile.tar',
                'dict_path': 'utils/dict/chinese_cht_dict.txt',
                
            },
            'it': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_it_mobile.tar',
                'dict_path': 'utils/dict/it_dict.txt',
                
            },
            'es': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_xi_mobile.tar',
                'dict_path': 'utils/dict/xi_dict.txt',
                
            },
            'pt': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_pu_mobile.tar',
                'dict_path': 'utils/dict/pu_dict.txt',
                
            },
            'ru': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_ru_mobile.tar',
                'dict_path': 'utils/dict/ru_dict.txt',
                
            },
            'ar': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_ar_mobile.tar',
                'dict_path': 'utils/dict/ar_dict.txt',
                
            },
            'hi': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_hi_mobile.tar',
                'dict_path': 'utils/dict/hi_dict.txt',
                
            },
            'ug': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_ug_mobile.tar',
                'dict_path': 'utils/dict/ug_dict.txt',
                
            },
            'fa': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_fa_mobile.tar',
                'dict_path': 'utils/dict/fa_dict.txt',
                
            },
            'ur': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_ur_mobile.tar',
                'dict_path': 'utils/dict/ur_dict.txt',
                
            },
            'rs_latin': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_rs_mobile.tar',
                'dict_path': 'utils/dict/rs_dict.txt',
                
            },
            'oc': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_oc_mobile.tar',
                'dict_path': 'utils/dict/oc_dict.txt',
                
            },
            'mr': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_mr_mobile.tar',
                'dict_path': 'utils/dict/mr_dict.txt',
                
            },
            'ne': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_ne_mobile.tar',
                'dict_path': 'utils/dict/ne_dict.txt',
                
            },
            'rs_cyrillic': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_rsc_mobile.tar',
                'dict_path': 'utils/dict/rsc_dict.txt',
                
            },
            'bg': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_bg_mobile.tar',
                'dict_path': 'utils/dict/bg_dict.txt',
                
            },
            'uk': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_uk_mobile.tar',
                'dict_path': 'utils/dict/uk_dict.txt',
                
            },
            'be': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_be_mobile.tar',
                'dict_path': 'utils/dict/be_dict.txt',
                
            },
            'te': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_te_mobile.tar',
                'dict_path': 'utils/dict/te_dict.txt',
                
            },
            'kn': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_ka_mobile.tar',
                'dict_path': 'utils/dict/ka_dict.txt',
                
            },
            'ta': {
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/rec_ta_mobile.tar',
                'dict_path': 'utils/dict/ta_dict.txt',
                
            }
        },
        'cls': {
            'url':
            'https://docyard.s3.us-west-000.backblazeb2.com/UCR_models/torch_mobile/cls_ench_ppocr_mobile.tar'
        }
    }
}


SUPPORT_DET_MODEL = ['DB', 'CRAFT']
SUPPORT_MODEL = ['CRNN']
VERSION = 1.0
SUPPORT_MODEL_TYPE = ['torch_mobile', 'torch_server', 'onnx_mobile', 'onnx_server']
BASE_DIR = os.path.expanduser("~/.ucr/")


def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes:
        logger.error("Something went wrong while downloading model.")
        sys.exit(0)


def maybe_download(model_storage_directory, url):
    # using custom model
    
    fname = url.split('/')[-1][0:-4]
    tmp_path = os.path.join(model_storage_directory, fname)
    if not os.path.exists(
            os.path.join(tmp_path, fname+'.pt')
    ) or not os.path.exists(
            os.path.join(tmp_path, fname+'.yml')):
    
        logger.info('Downloading {} to {}'.format(url, tmp_path))
        os.makedirs(tmp_path, exist_ok=True)
        download_with_progressbar(url, tmp_path+'.tar')
        
        with tarfile.open(tmp_path+'.tar', 'r') as tarObj:
            tarObj.extractall(path=tmp_path, members=tarObj)
        os.remove(tmp_path+'.tar')
        
    return [os.path.join(tmp_path, fname+'.pt'), os.path.join(tmp_path, fname+'.yml')]

def parse_args(mMain=True, add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    if mMain:
        parser = argparse.ArgumentParser()
        # params for prediction engine
        parser.add_argument("--device", type=str, default='cpu')

        # params for text detector
        parser.add_argument("--image_location", type=str)
        parser.add_argument("--det_algorithm", type=str, default='DB')
        parser.add_argument("--det_model_location", type=str, default=None)
        parser.add_argument("--det_config_location", type=str, default=None)
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
        parser.add_argument("--rec_model_location", type=str, default=None)
        parser.add_argument("--rec_config_location", type=str, default=None)
        parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
        parser.add_argument("--rec_char_type", type=str, default='ch_sim')
        parser.add_argument("--rec_whitelist", type=str, default='') #whitelist characters while prediction
        parser.add_argument("--rec_blacklist", type=str, default='') #blacklist characters while prediction    
        parser.add_argument("--rec_batch_num", type=int, default=8)
        parser.add_argument("--max_text_length", type=int, default=25)
        parser.add_argument(
            "--rec_char_dict_path",
            type=str,
            default="utils/dict/ench_dict.txt")
        parser.add_argument("--use_space_char", type=str2bool, default=True)
        parser.add_argument(
            "--vis_font_path", type=str, default="utils/fonts/only_en.ttf")
        parser.add_argument("--drop_score", type=float, default=0.)

        # params for text classifier
        parser.add_argument("--cls_model_location", type=str, default=None)
        parser.add_argument("--cls_config_location", type=str, default=None)
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
        
        parser.add_argument("--lang", type=str, default='ch_sim')
        parser.add_argument("--type", type=str, default='mobile')
        parser.add_argument("--backend", type=str, default='torch')
        parser.add_argument("--det", type=str2bool, default=True)
        parser.add_argument("--rec", type=str2bool, default=True)
        parser.add_argument("--cls", type=str2bool, default=False)
                        
        return parser.parse_args()
    
    else:
        return argparse.Namespace(
            device='cpu',
            image_location='',
            det_algorithm='DB',
            max_batch_size=1,
            det_model_location=None,
            det_config_location=None,
            det_limit_side_len=1920,
            det_limit_type='min',
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            det_east_score_thresh=0.8,
            det_east_cover_thresh=0.1,
            det_east_nms_thresh=0.2,
            det_craft_text_thresh=0.35,
            det_craft_min_size=3,
            det_craft_link_thresh=0.1,
            det_craft_rotated_box=True,
            det_craft_use_dilate=False,
            det_craft_xdilate=9,
            det_craft_ydilate=3,
            det_craft_xpad=4,
            det_craft_ypad=2,
            rec_algorithm='CRNN',
            rec_model_location=None,
            rec_config_location=None,
            rec_image_shape="3, 32, 320",
            rec_char_type='ch_sim',
            rec_batch_num=30,
            rec_whitelist='',
            rec_blacklist='',
            max_text_length=25,
            rec_char_dict_path=None,
            use_space_char=True,
            drop_score=0.,
            cls_model_location=None,
            cls_config_location=None,
            cls_image_shape="3, 48, 192",
            label_list=['0', '180'],
            cls_batch_num=30,
            cls_thresh=0.9,
            vis_font_path='utils/fonts/only_en.ttf',
            merge_boxes = False,
            merge_slope_thresh = 0.1, 
            merge_ycenter_thresh = 0.8, 
            merge_height_thresh = 0.6,
            merge_width_thresh = 2.0,
            merge_add_margin = 0.05,
            lang='ch_sim',
            type='mobile',
            backend='torch',
            det=True,
            rec=True,
            cls=False)


class UCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        postprocess_params = parse_args(mMain=False, add_help=False)
        postprocess_params.__dict__.update(**kwargs)
        lang = postprocess_params.lang
        type = postprocess_params.type
        backend = postprocess_params.backend
        model_type = backend + '_' + type
        det_algorithm = postprocess_params.det_algorithm
        rec_algorithm = postprocess_params.rec_algorithm
        self.det = postprocess_params.det
        self.rec = postprocess_params.rec
        self.cls = postprocess_params.cls
        if det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if rec_algorithm not in SUPPORT_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_MODEL))
            sys.exit(0)
        
        assert model_type in SUPPORT_MODEL_TYPE, 'TYPE_BACKEND must be of {} format, but got {}'.format(
                SUPPORT_MODEL_TYPE, model_type)

            
        if postprocess_params.rec_char_dict_path is None:            
            model_type = 'torch_mobile' if lang != 'ch_sim' else model_type
            postprocess_params.rec_char_dict_path = model_urls[model_type]['rec'][lang][
                'dict_path']
            model_type = backend + '_' + type
        
        # download model
        model_type = 'torch_server' if det_algorithm == 'CRAFT' else model_type
        det_path, det_cfg_path = maybe_download(os.path.join(BASE_DIR, '{}/det'.format(VERSION)), model_urls[model_type]['det'][det_algorithm]['url'])
        model_type = backend + '_' + type
        
        model_type = 'torch_mobile' if lang != 'ch_sim' else model_type
        rec_path, rec_cfg_path = maybe_download(os.path.join(BASE_DIR, '{}/rec/{}'.format(VERSION, lang)),
                       model_urls[model_type]['rec'][lang]['url'])
        model_type = backend + '_' + type
        
        cls_path, cls_cfg_path = maybe_download(os.path.join(BASE_DIR, '{}/cls'.format(VERSION)), model_urls['torch_mobile']['cls']['url'])
        
        # init model dir
        if postprocess_params.det_model_location is None:
            postprocess_params.det_model_location = det_path
            postprocess_params.det_config_location = det_cfg_path
        if postprocess_params.rec_model_location is None:
            postprocess_params.rec_model_location = rec_path
            postprocess_params.rec_config_location = rec_cfg_path
        if postprocess_params.cls_model_location is None:
            postprocess_params.cls_model_location = cls_path
            postprocess_params.cls_config_location = cls_cfg_path
        
        
        postprocess_params.rec_char_dict_path = str(
            Path(__file__).parent / postprocess_params.rec_char_dict_path)
        
        super().__init__(postprocess_params)

    def predict(self, img, det=None, rec=None, cls=None):
        """
        ocr with paddleocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not, if false, only rec will be exec. default is True
            rec: use text recognition or not, if false, only det will be exec. default is True
        """
        if det==None:
            det = self.det
        if rec==None:
            rec = self.rec
        if cls==None:
            cls = self.cls
        assert isinstance(img, (np.ndarray, list, str))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)

        if isinstance(img, str):
            # download net image
            if img.startswith('http'):
                download_with_progressbar(img, 'tmp.jpg')
                img = 'tmp.jpg'
            image_file = img
            img, flag = check_and_read_gif(image_file)
            if not flag:
                with open(image_file, 'rb') as f:
                    np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if det and rec:
            dt_boxes, rec_res = self.__call__(img)
            return [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        elif det and not rec:
            dt_boxes, elapse = self.text_detector(img)
            if dt_boxes is None:
                return None
            return [box.tolist() for box in dt_boxes]
        else:
            if not isinstance(img, list):
                img = [img]
            if cls:
                img, cls_res, elapse = self.text_classifier(img)
                if not rec:
                    return cls_res
            rec_res, elapse = self.text_recognizer(img)
            return rec_res


def main():
    # for cmd
    args = parse_args(mMain=True)
    image_location = args.image_location
    if image_location.startswith('http'):
        download_with_progressbar(image_location, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(args.image_location)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(args.image_location))
        return

    ocr_engine = UCR(**(args.__dict__))
    for img_path in image_file_list:
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
        result = ocr_engine.predict(img_path,
                                det=args.det,
                                rec=args.rec,
                                cls=args.cls)
        if result is not None:
            for line in result:
                logger.info(line)
