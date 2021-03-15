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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .iaa_augment import IaaAugment
from .make_border_map import MakeBorderMap
from .make_shrink_map import MakeShrinkMap
from .random_crop_data import EastRandomCropData, PSERandomCrop

from .rec_img_aug import RecAug, RecResizeImg, ClsResizeImg
from .randaugment import RandAugment
from .operators import *
from .label_ops import *

from .east_process import *
from .sast_process import *

import copy

__all__ = ['build_preprocess', 'preprocess']


def build_preprocess(config, global_config=None):
    
    config = copy.deepcopy(config)
    assert isinstance(config, list), ('operator config should be a list')
    ops = []
    for preprocess in config:
        assert isinstance(preprocess,
                          dict) and len(preprocess) == 1, "yaml format error"
        
        op_name = list(preprocess)[0]
        param = {} if preprocess[op_name] is None else preprocess[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
        
    return ops

def preprocess(data, ops=None):
    """ preprocess """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data
