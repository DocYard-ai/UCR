<br> <br>
<p align="center"><img src="docs/static/images/VectorU.svg" alt="Github Runner Covergae Status" height="100">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="docs/static/images/VectorC.svg" alt="Github Runner Covergae Status" height="100">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="docs/static/images/VectorR.svg" alt="Github Runner Covergae Status" height="100"></p>
<br> <br>
<p align="center">Universal Character Recognizer (UCR) is an <u>Open Source</u>, <u>Easy to use</u> Python library to build <u>Production Ready</u> OCR applications with its highly Intuitive,  Modular & Extensible API design and off-the-shelf <a href="docs/modelzoo.md">Pretrained Models</a> for over <b>25 languages</b>.</p>
<p align="center">
  Read UCR Documentation on <u><a href="https://ucr.docyard.ai/">ucr.docyard.ai</a></u>
  <br> <br>
  <a href="#about">Features</a> •
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#acknowledgement">Acknowledgement</a>
  <br> <br>
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/ucr">
  <a href="https://badge.fury.io/py/ucr"><img src="https://badge.fury.io/py/ucr.svg" alt="PyPI version"></a>
</p>

## Demo

#### For details, click [here](https://ucr.docyard.ai/demo)!

<p align="center"><img src="docs/static/images/demo.gif"/></p>

## Features

- Supports SOTA Text Detection and Recognition models
- Built on top of Pytorch and Pytorch Lightning
- Supports over 25 languages
- Model Zoo contains 27 Pretrained Models across 25 languages
- Modular Design Language allows Pick and Choose of different components
- Easily extensible with Custom Components and attributes
- Hydra config enables Rapid Prototyping with multiple configurations
- Support for Packaging, Logging and Deployment tools straight out of the box

*Note: Some features are still in active development and might not be available.*
## Setup

### Installation

**Require python version >= 3.6.2, install with `pip` (recommended)**

1. <b>Prerequisites:</b> Install compatible version of Pytorch and torchvision from [official repository](https://pytorch.org/get-started/locally/).
2. <b>Installation:</b> Install the latest stable version of UCR:
```shell
pip install -U ucr
```

#### <span style="color:#FF8856">[Optional]</span> Test Installation

Run dummy tests!
```python
ucr test
# Optional: Add -l/--lang='language_id' to test on particular language!
ucr test -l='en_number'
```  


## Usage
### Workflow


<p align="center"><img src="docs/static/images/workflow.png"/></p>

Execution flow of UCR is displayed above. Broadly it can be divided into 4 sub-parts:

1. Input (image/folder path or web address) is fed into the <u>Detection</u> model which outputs bounding box coordinates of all the text boxes.
2. The detected boxes are then checked for <u>Orientation</u> and corrected accordingly.
3. Next, <u>Recognition</u> model runs on the corrected text boxes. It returns bounding box information and OCR output.
4. Lastly, an optional <u>Post Processing</u> module is executed to improve/modify the results.

### Quick Start

The following code snippet shows how to get started with UCR library.

```python
from ucr import UCR

# initialization
ocr = UCR(lang="en_number", device="cpu")

# run prediction
result = ocr.predict('input_path', output='output_path')

# for saving annotated image
result = ocr.predict('input_path', output='output_path', save_image=True)
```
For complete list of arguments, refer <a href="docs/tldr.md/#argument-list">Argument List</a>

## Model Zoo

A collection of pretrained models for detection, classification and recognition processes is present <a href="docs/modelzoo.md">here</a> !  
These models can be useful for out-of-the-box inference on over 25 languages.


## Acknowledgement

Substantial part of the UCR library is either inspired or inherited from the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) library. Wherever possible the repository has been ported from PaddlePaddle to PyTorch framework including the direct translation of model parameters.
Also, a big thanks to [Clova AI](https://clova.ai/en/research/research-areas.html), for open sourcing their testing script and pretrained models ([CRAFT](https://github.com/clovaai/CRAFT-pytorch)).  

## License

[Apache License 2.0](LICENSE)
