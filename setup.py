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

from setuptools import setup

setup(
    name='ucr',
    packages=['ucr'],
    package_dir={'ucr': './'},
    include_package_data=True,
    entry_points={"console_scripts": ["ucr= ucr.ucr:main"]},
    version='0.1.2',
    author='DocYardAI',
    install_requires=[
        # Do not add opencv here. Just like pytorch, user should install
        # opencv themselves, preferrably by OS's package manager, or by
        # choosing the proper pypi package name at https://github.com/skvark/opencv-python
        "shapely",
        "scikit-image>=0.17.2",
        "imgaug>=0.4.0",
        "pyclipper",
        "lmdb",
        "opencv-python>=4.2.0.32",
        "tqdm",
        "numpy",
        "visualdl",
        "python-Levenshtein",
    ],
    license='Apache License 2.0',
    description='Universal Character Recognizer (UCR): Simple, Intuitive, Extensible, Multi-Lingual OCR engine',
    url='https://github.com/DocYard/UCR',
    download_url='https://github.com/DocYard/UCR.git',
    keywords=[
        'opticalcharacterrecognition textdetection textrecognition ocr ucr crnn east star-net rosetta db craft englishocr chineseocr'
    ],
)
