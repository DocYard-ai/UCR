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

import copy
from torch.utils.data import DataLoader

__all__ = ['build_dataloader']


def build_dataloader(config, mode, logger):
    config = copy.deepcopy(config)

    support_dict = ['SimpleDataSet', 'LMDBDateSet']
    module_name = config[mode]['Dataloader']['name']
    assert module_name in support_dict, Exception(
        'DataSet only support {}'.format(support_dict))
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."

    dataloader = eval(module_name)(config, mode, logger)
    loader_config = config[mode]['Dataloader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']

    data_loader = DataLoader(
        dataloader=dataloader,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last)

    return data_loader
