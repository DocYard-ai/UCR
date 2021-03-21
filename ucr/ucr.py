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
import pandas as pd
import requests
from tqdm import tqdm
import shutil
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))

from omegaconf import OmegaConf
from hydra.experimental import compose, initialize_config_dir

from ucr.inference import infer_system
from ucr.utils.utility import check_and_read_gif, get_image_file_list
from ucr.utils.annotation import draw_ocr_box_txt, draw_text_det_res

__all__ = ['UCR']

SUPPORT_DET_MODEL = ['DB', 'CRAFT']
SUPPORT_REC_MODEL = ['CRNN']
SUPPORT_CLS_MODEL = ['CLS']
VERSION = 1.0
SUPPORT_MODEL_TYPE = ['torch_mobile', 'torch_server', 'onnx_mobile', 'onnx_server']
BASE_DIR = os.path.expanduser("~/.ucr/")


class UCR(infer_system.TextSystem):
    def __init__(self, **kwargs):
        """
        ucr package
        args:
            **kwargs: other params show in ucr --help
        """
        args = parse_args(mMain=False, add_help=False)
        args.__dict__.update(**kwargs)
        
        force_download = args.force_download
        conf_location = args.hydra_conf_location
        det_algorithm = args.det_algorithm
        rec_algorithm = args.rec_algorithm
        cls_algorithm = args.cls_algorithm

        type = args.type
        backend = args.backend
        model_type = backend + '_' + type
        
        lang = args.l
        if args.lang!='ch_sim': 
            lang = args.lang
            if args.l!='ch_sim':
                print(f'WARNING: "l (={args.l})" argument is superseded by "lang (={args.lang})" argument! l=lang={lang}')

        device = args.d
        if args.device!='cpu':
            device = args.device
            if args.d!='cpu':
                print(f'WARNING: "d (={args.d})" argument is superseded by "device (={args.device})" argument! d=device={device}')

        if not conf_location:
            conf_location = maybe_download(os.path.join(BASE_DIR, '{}/conf'.format(VERSION)),model_urls['conf']['url'], force_download)
        
        with initialize_config_dir(config_dir=conf_location, job_name="infer_det"):
            model_type = 'torch_server' if det_algorithm == 'CRAFT' else model_type
            overrides = [f"{k}={v}" for k, v in model_urls[model_type]['det'][det_algorithm].items() if k!='url']
            overrides.extend(args.det_overrides)
            cfg = compose(config_name=args.det_config_name, overrides=overrides)    
            # print("Detection config:\n{}\n".format(cfg.pretty()))    
            
            config_det = OmegaConf.to_container(cfg)
            model_type = backend + '_' + type
            
        with initialize_config_dir(config_dir=conf_location, job_name="infer_rec"):
            model_type = 'torch_mobile' if lang != 'ch_sim' else model_type
            overrides = [f"{k}={v}" for k, v in model_urls[model_type]['rec'][lang].items() if k!='url']
            overrides.extend(args.rec_overrides)
            cfg = compose(config_name=args.rec_config_name, overrides=overrides)      
            # print("Recognition config:\n{}\n".format(cfg.pretty()))    
                  
            config_rec = OmegaConf.to_container(cfg)
            model_type = backend + '_' + type
            
        with initialize_config_dir(config_dir=conf_location, job_name="infer_cls"):
            overrides = [f"{k}={v}" for k, v in model_urls['torch_mobile']['cls'].items() if k!='url']
            overrides.extend(args.cls_overrides)
            cfg = compose(config_name=args.cls_config_name, overrides=overrides)    
            # print("Classification config:\n{}\n".format(cfg.pretty()))    
                    
            config_cls = OmegaConf.to_container(cfg)
        
        with initialize_config_dir(config_dir=conf_location, job_name="infer_system"):
            cfg = compose(config_name="infer_system", overrides=args.system_overrides)    
            # print("End-to-end config:\n{}\n".format(cfg.pretty()))    
                    
            config = OmegaConf.to_container(cfg)
            
        assert model_type in SUPPORT_MODEL_TYPE, 'TYPE_BACKEND must be of {} format, but got {}'.format(
                SUPPORT_MODEL_TYPE, model_type)
        
        config_det['device'] = config_rec['device'] = config_cls['device'] = device
        
        config_det['batch_size'] = args.det_batch_size
        config_rec['batch_size'] = args.rec_batch_size
        config_cls['batch_size'] = args.cls_batch_size
        
        config_rec['whitelist'] = args.rec_whitelist
        config_rec['blacklist'] = args.rec_blacklist
        config_rec['lang'] = lang
        
        # init model dir
        if not config_det['model_location']:
            model_type = 'torch_server' if det_algorithm == 'CRAFT' else model_type
            det_path = maybe_download(os.path.join(BASE_DIR, '{}/det'.format(VERSION)), model_urls[model_type]['det'][det_algorithm]['url'], force_download)
            config_det['model_location'] = os.path.join(det_path, 'model.pt')
            model_type = backend + '_' + type
        else:
            config_det['model_location'] = config_det['model_location'] # add abs path in parser args. 
        
        if not config_rec['model_location']:
            model_type = 'torch_mobile' if lang != 'ch_sim' else model_type
            rec_path = maybe_download(os.path.join(BASE_DIR, '{}/rec/{}'.format(VERSION, lang)), model_urls[model_type]['rec'][lang]['url'], force_download)
            config_rec['model_location'] = os.path.join(rec_path, 'model.pt')   
            
            __dir__ = os.path.dirname(__file__)
            config_rec['font_path'] = os.path.join(__dir__, config_rec['font_path'])  # TODO: Zip both font_path and char_dict_location in model_url
            config_rec['char_dict_location'] = os.path.join(__dir__, config_rec['char_dict_location'])   
        else:
            config_rec['model_location'] = config_rec['model_location'] # add abs path in parser args. 
            config_rec['font_path'] = config_rec['font_path'] # add abs path in .yaml file  
            config_rec['char_dict_location'] = config_rec['char_dict_location'] # add abs path in .yaml file       
            
        if not config_cls['model_location']:
            cls_path = maybe_download(os.path.join(BASE_DIR, '{}/cls'.format(VERSION)), model_urls['torch_mobile']['cls']['url'], force_download)
            config_cls['model_location'] = os.path.join(cls_path, 'model.pt')  
        else:
            config_cls['model_location'] = config_cls['model_location'] # add abs path in parser args.

        config['Detection'] = config_det
        config['Recognition'] = config_rec
        config['Classification'] = config_cls
        
        self.merge_boxes = config["merge_boxes"]
        self.drop_score = config["drop_score"]
        self.font_path = config_rec['font_path']
        self.is_visualize = config["is_visualize"]
        self.output_format = args.output_format
        
        super().__init__(config)
            
    def perform_ocr(self, img, key, output, det, rec, cls):
        
        if det and rec:
            dt_boxes, rec_res = self.__call__(img, cls)
            if self.merge_boxes:
                rec_res = rec_res[0:1]
            
            if self.output_format=='ppocr':
                value = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            elif self.output_format=='df':
                info_list = [[int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1]), res[0]] for box, res in zip(dt_boxes, rec_res)]
                value = pd.DataFrame(info_list, columns=['startX', 'startY', 'endX', 'endY', 'OCR'])
                
            if self.is_visualize:
                from tabulate import tabulate
                headers = ["OCR Result", "Score"]
                if self.merge_boxes:
                    print(tabulate(rec_res[0:1], headers, tablefmt="fancy_grid"))
                else:
                    print(tabulate(rec_res, headers, tablefmt="fancy_grid"))
                    
            if output:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=self.drop_score,
                    font_path=self.font_path)
                
                if not os.path.exists(output):
                    os.makedirs(output)
                img_path = os.path.join(output,
                                        "ocr_{}".format(key))
                cv2.imwrite(
                    img_path, draw_img[:, :, ::-1])
                print("Saving OCR ouput in {}".format(
                    img_path))
            
            return value
            
        elif det and not rec:
            dt_boxes, elapse = self.text_detector(img)
            if self.is_visualize:
                print("\n------------------------dt_boxes num : {},\telapse : {:.3f}------------------------".format(
                len(dt_boxes), elapse))
            if dt_boxes is not None:
                if self.output_format=='ppocr':
                    value = [box.tolist() for box in dt_boxes]
                    
                elif self.output_format=='df':
                    info_list = [[int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])] for box in dt_boxes]
                    value = pd.DataFrame(info_list, columns=['startX', 'startY', 'endX', 'endY'])
                
                if output:
                    src_im = draw_text_det_res(dt_boxes, img)
                    
                    if not os.path.exists(output):
                        os.makedirs(output)
                    img_path = os.path.join(output,
                                            "det_{}".format(key))
                    cv2.imwrite(img_path, src_im)
                    print("Detection output image is saved in {}".format(img_path))
            
            return value
            
        else:
            return img 
    
    def predict(self, input=None, i=None, output=None, o=None, det=True, rec=True, cls=False):
        """
        ocr with paddleocr
        args：
            input: input for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not, if false, only rec will be exec. default is True
            rec: use text recognition or not, if false, only det will be exec. default is True
        """
        
        if input is not None:
            input = input
        elif i is not None:
            input = i
        elif input is None and i is None:
            print('Input is mandatory: can be either ndarray (or list of ndarray), img_filepath, img_folderpath or img_webpath!')
            sys.exit(0)
        
        assert isinstance(input, (np.ndarray, list, str))

        rec_dict = {}
        out_dict={} 
        
        if output is not None:
            output = output
        elif o is not None:
            output = o
            
        if output:
            print("Saving prediction results in '{}' folder\n".format(output))  
            if not os.path.exists(output):
                os.makedirs(output)   
        
        is_imgpath = False
        if isinstance(input, str):
            # download net image
            if input.startswith('http') and input.endswith('.jpg'):
                download_with_progressbar(input, 'downloaded.jpg')
                input = 'downloaded.jpg'
                
            input = get_image_file_list(input)
            
        elif isinstance(input, np.ndarray):
            input = [input]
        
        else:
            if isinstance(input[0], str):
                is_imgpath = True
                
        if len(input) == 0:
            print('NO images found in {}. Please check input location!'.format(input))
            sys.exit(0) 
        
        i = 0
        for image in tqdm(input, colour='green', desc='OCR'):
            if is_imgpath:
                key = os.path.basename(image)
                img, flag = check_and_read_gif(image)
                if not flag:
                    img = cv2.imread(image)
                if img is None:
                    print("ERROR in loading image:{}".format(image))
                    continue
            else:
                key = str(i) + '.jpg'
                i +=1
                if len(image.shape) == 2:
                    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
                elif len(image.shape) == 3:
                    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
                elif len(image.shape) == 4: # png file
                    img = cv2.cvtColor(image[:3], cv2.COLOR_RGB2BGR) 
                else:
                    print("ERROR: Input array has {} channels. Expected '2', '3' or '4'.".format(len(image.shape)))
                    continue  
            result = self.perform_ocr(img, key, output, det, rec, cls)
            if det:
                out_dict[key] = result
            else:
                rec_dict[key] = result
                    
        if rec_dict:
            img = list(rec_dict.values())
            if cls:
                img, cls_res, elapse = self.text_classifier(img)
                if not rec:
                    if self.is_visualize:
                        cls_list = [[k,v[0], v[1]] for k,v in zip(list(rec_dict.keys()), cls_res)]
                        print("Classification Result:\n")
                        from tabulate import tabulate
                        headers = ["File Name", "Classification Result", "Score"]
                        print(tabulate(cls_list, headers, tablefmt="fancy_grid"))
                        
                    if output:
                        output_names = [f'{k}_{v[0]}.jpg' for k,v in zip(list(rec_dict.keys()), cls_res)]    
                        for bno in range(len(img)):
                            cv2.imwrite(os.path.join(output,output_names[bno]), img[bno])
                        
                    cls_output = {k:[v] for k,v in zip(list(rec_dict.keys()), cls_res)}       
                    return cls_output               
            if rec:    
                rec_res, elapse = self.text_recognizer(img)
                if self.is_visualize:
                    rec_list = [[k,v[0], v[1]] for k,v in zip(list(rec_dict.keys()), rec_res)]
                    print("Recognition Result:\n")
                    from tabulate import tabulate
                    headers = ["File Name", "Recognition Result", "Score"]
                    print(tabulate(rec_list, headers, tablefmt="fancy_grid"))
                                            
                if output:
                    output_names = [f'{k}_{v[0]}.jpg' for k,v in zip(list(rec_dict.keys()), rec_res)]    
                    for bno in range(len(img)):
                        cv2.imwrite(os.path.join(output,output_names[bno]), img[bno])
                        
                rec_output = {k:[v] for k,v in zip(list(rec_dict.keys()), rec_res)} 
                return rec_output
        else:
            return out_dict


def main():
    args = parse_args(mMain=True)
    
    ocr_engine = UCR(**(args.__dict__))
    result = ocr_engine.predict(input=args.input,
                            output=args.output,
                            det=args.det,
                            rec=args.rec,
                            cls=args.cls)
    if result is not None:
        for k,v in result.items():
            print(f'\n------------------------{k}:------------------------')
            from tabulate import tabulate
            print(tabulate(v, tablefmt="fancy_grid"))
            

def download_with_progressbar(url, save_path):
    print("Downloading '{}' to '{}'".format(url, save_path))
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
        print("Something went wrong while downloading model.")
        sys.exit(0)


def maybe_download(model_storage_directory, url, force_download=False):
    # using custom model
    
    fname = url.split('/')[-1][0:-4]
    tmp_path = os.path.join(model_storage_directory, fname)
    if force_download or not os.path.exists(tmp_path):
        os.makedirs(tmp_path, exist_ok=True)
        download_with_progressbar(url, tmp_path+'.zip')
        
        shutil.unpack_archive(tmp_path+'.zip', tmp_path, 'zip')
        
        os.remove(tmp_path+'.zip')
        
    return tmp_path

def parse_args(mMain=True, add_help=True):
    import argparse
    
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    if mMain:
        parser = argparse.ArgumentParser()
        
        # params for prediction system
        parser.add_argument("--hydra_conf_location", type=str, default=None)
        parser.add_argument("--force_download", type=str2bool, default=False)
        parser.add_argument("-i", "--input", type=str, required=True)
        parser.add_argument("-o", "--output", type=str, default=None)
        parser.add_argument("-d", "--device", type=str, default='cpu')
        parser.add_argument("-l", "--lang", type=str, default='ch_sim')
        parser.add_argument("--backend", type=str, default='torch')
        parser.add_argument("--type", type=str, default='mobile')
        parser.add_argument("--output_format", type=str, default='df')
        parser.add_argument("--system_overrides", nargs='+', type=str, default=[])
        parser.add_argument("--det", type=str2bool, default=True)
        parser.add_argument("--rec", type=str2bool, default=True)
        parser.add_argument("--cls", type=str2bool, default=False)
        
        
        # params for detection engine
        parser.add_argument("--det_algorithm", type=str, default='CRAFT')
        parser.add_argument("--det_config_name", type=str, default='infer_det')
        parser.add_argument("--det_model_location", type=str, default=None)
        parser.add_argument("--det_batch_size", type=int, default=1) #batch_size is not yet implemented in the code
        parser.add_argument("--det_overrides", nargs='+', type=str, default=[])
        
        # params for recogniton engine
        parser.add_argument("--rec_algorithm", type=str, default='CRNN')
        parser.add_argument("--rec_config_name", type=str, default='infer_rec')
        parser.add_argument("--rec_model_location", type=str, default=None)
        parser.add_argument("--rec_batch_size", type=int, default=8) 
        parser.add_argument("--rec_overrides", nargs='+', type=str, default=[])
        parser.add_argument("--rec_whitelist", type=str, default=None)
        parser.add_argument("--rec_blacklist", type=str, default=None)
        
        # params for detection engine
        parser.add_argument("--cls_algorithm", type=str, default='CLS')
        parser.add_argument("--cls_config_name", type=str, default='infer_cls')
        parser.add_argument("--cls_model_location", type=str, default=None)
        parser.add_argument("--cls_batch_size", type=int, default=8)
        parser.add_argument("--cls_overrides", nargs='+', type=str, default=[])

        return parser.parse_args()
    else:
        return argparse.Namespace(
            hydra_conf_location=None,
            force_download=False,
            d = 'cpu',
            device='cpu',
            l = 'ch_sim',
            lang='ch_sim',
            type='mobile',
            backend='torch',
            output_format='df',
            system_overrides=[],
            
            det_algorithm='CRAFT',
            det_config_name='infer_det',
            det_model_location=None,
            det_batch_size=1,
            det_overrides=[],
            
            rec_algorithm='CRNN',
            rec_config_name='infer_rec',
            rec_model_location=None,
            rec_batch_size=8,
            rec_overrides=[],
            rec_whitelist=None,
            rec_blacklist=None,
            
            cls_algorithm='CLS',
            cls_config_name='infer_cls',
            cls_model_location=None,
            cls_batch_size=8,
            cls_overrides=[],
        )


model_urls = {
    
    'conf': {
        'url': 'https://docyard.s3.us-west-000.backblazeb2.com/UCR/conf/conf.zip'
    },
    
    'torch_server': {
        'det': {
            'DB': {
                'preprocess': 'det_db',
                'architecture': 'det_r50_vd_db',
                'postprocess': 'det_db',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_server/det_ench_ppocr_server.zip'
            },
            'CRAFT': {
                'preprocess': 'det_craft',
                'architecture': 'det_vgg_craft',
                'postprocess': 'det_craft',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_server/det_vgg_craft.zip'
            }    
        },
        'rec': {
            'ch_sim': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_server',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/simfang.ttf',
                'char_dict_location': 'utils/dict/ench_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_server/rec_ench_ppocr_server.zip',                
            }
        }
    },

    'torch_mobile': {
        'det': {
            'DB': {
                'preprocess': 'det_db',
                'architecture': 'det_mv3_db',
                'postprocess': 'det_db',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/det_ench_ppocr_mobile.zip',
            }  
        },
        'rec': {
            'ch_sim': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/simfang.ttf',
                'char_dict_location': 'utils/dict/ench_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_ench_ppocr_mobile.zip',                
            },
            'en': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/only_en.ttf',
                'char_dict_location': 'utils/dict/en_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_en_number_mobile.zip',                
            },
            'fr': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/french.ttf',
                'char_dict_location': 'utils/dict/french_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_french_mobile.zip',                
            },
            'de': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/german.ttf',
                'char_dict_location': 'utils/dict/german_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_german_mobile.zip',                
            },
            'ko': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/korean.ttf',
                'char_dict_location': 'utils/dict/korean_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_korean_mobile.zip',                
            },
            'ja': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/japan.ttc',
                'char_dict_location': 'utils/dict/japan_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_japan_mobile.zip',                
            },
            'ch_tra': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/chinese_cht.ttf',
                'char_dict_location': 'utils/dict/chinese_cht_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_chinese_cht_mobile.zip',                
            },
            'it': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/italian.ttf',
                'char_dict_location': 'utils/dict/it_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_it_mobile.zip',                
            },
            'es': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/xi.ttf',
                'char_dict_location': 'utils/dict/spanish_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_xi_mobile.zip',                
            },
            'pt': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/portugese.ttf',
                'char_dict_location': 'utils/dict/pu_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_pu_mobile.zip',                
            },
            'ru': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/russian.ttf',
                'char_dict_location': 'utils/dict/ru_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_ru_mobile.zip',                
            },
            'ar': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/arabic.ttf',
                'char_dict_location': 'utils/dict/ar_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_ar_mobile.zip',                
            },
            'hi': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/hindi.ttf',
                'char_dict_location': 'utils/dict/hi_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_hi_mobile.zip',                
            },
            'ug': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/uyghur.ttf',
                'char_dict_location': 'utils/dict/ug_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_ug_mobile.zip',                
            },
            'fa': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/persian.ttf',
                'char_dict_location': 'utils/dict/fa_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_fa_mobile.zip',                
            },
            'ur': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/urdu.ttf',
                'char_dict_location': 'utils/dict/ur_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_ur_mobile.zip',                
            },
            'rs_latin': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/serbian.ttf',
                'char_dict_location': 'utils/dict/rs_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_rs_mobile.zip',                
            },
            'oc': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/serbian.ttf',
                'char_dict_location': 'utils/dict/oc_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_oc_mobile.zip',                
            },
            'mr': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/marathi.ttf',
                'char_dict_location': 'utils/dict/mr_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_mr_mobile.zip',                
            },
            'ne': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/nepali.ttf',
                'char_dict_location': 'utils/dict/ne_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_ne_mobile.zip',                
            },
            'rs_cyrillic': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/serbian.ttf',
                'char_dict_location': 'utils/dict/rsc_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_rsc_mobile.zip',                
            },
            'bg': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/serbian.ttf',
                'char_dict_location': 'utils/dict/bg_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_bg_mobile.zip',                
            },
            'uk': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/serbian.ttf',
                'char_dict_location': 'utils/dict/uk_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_uk_mobile.zip',                
            },
            'be': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/serbian.ttf',
                'char_dict_location': 'utils/dict/be_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_be_mobile.zip',                
            },
            'te': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/telugu.ttf',
                'char_dict_location': 'utils/dict/te_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_te_mobile.zip',                
            },
            'kn': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/kannada.ttf',
                'char_dict_location': 'utils/dict/ka_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_ka_mobile.zip',                
            },
            'ta': {
                'preprocess': 'rec_ctc',
                'architecture': 'rec_ppocr_mobile',
                'postprocess': 'rec_ctc',
                'font_path': 'utils/fonts/tamil.ttf',
                'char_dict_location': 'utils/dict/ta_dict.txt',
                'url':
                'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/rec_ta_mobile.zip',                
            }
        },
        'cls': {
            'preprocess': 'cls',
            'architecture': 'cls_ppocr_mobile',
            'postprocess': 'cls',
            'url':
            'https://docyard.s3.us-west-000.backblazeb2.com/UCR/torch_mobile/cls_ench_ppocr_mobile.zip'
        }
    }
}
