!!! important "Important Notice"
    This is a **Quick Start** guide on how to use the UCR library with its existing pretrained models. For detailed information regarding its usage and advanced functionality, please refer to other sections!

## Set Up

1. <span style="color:#FF8856">Prerequisites:</span> Install compatible version of Pytorch and torchvision from [official repository](https://pytorch.org/get-started/locally/).
2. <span style="color:#FF8856">Installation:</span> Install the latest stable version of UCR:
```bash
pip install -U ucr
```
3. <span style="color:#FF8856">Test Installation (Optional):</span> Run dummy tests!
```bash
ucr test
```  

*For other installation modes and complete setup instructions, see [here](coming_soon.md)!*

## Run Prediction

1. <span style="color:#FF8856">Using Command Line:</span>
```bash
# run prediction on input file/folder
ucr -i input_path -o output_path -l language_id
```
List of all supported languages with their corresponding ids is shown [here](coming_soon.md)!  
To view all available options for the CLI application:
```bash
# view all CLI commands
ucr --help
```
Usage: ucr *--options* <span style="color:#FF8856">[ARGS]</span> 
1. <span style="color:#FF8856">From Python scripts/Jupyter Notebooks:</span>
```python
# Import and Initialize
from ucr import UCR
ocr = UCR(lang='en') #lang='language_id'.

# Run Predictions
result = ocr.predict(input='input_path', output='output_path')
```  

*Refer to the [Demo](coming_soon.md) page and/or [Colab Notebook](coming_soon.md) to see it in action!*

## Argument List 

|    <span style="font-weight:bold; font-size: 125%">Name</span>                  | <span style="font-weight:bold; font-size: 125%">Type</span>     | <span style="font-weight:bold; font-size: 125%">Default</span>   | <span style="font-weight:bold; font-size: 125%">Help</span> |
|----------------------|----------|-----------|------|
| `i/input`            | str/array/list      | <span style="font-style: italic; color:#FF8856">Required</span>          |Path to input location, eg: file/folder path, web address, numpy array etc. More details [here](coming_soon.md)!|
| `o/output`           | str      | None      | Path to output folder. <span style="font-style: italic; color:#FF8856">(optional)</span>|
| `l/lang`             | str      | "ch_sim"    | List of supported language-ids can be found [here](coming_soon.md)!     |
| `d/device`           | str      | "cuda"      | Specify device to run on. *["cuda", "cpu"]*      |
| `verbose`            | bool     | False     | Enable it to print info on console!     |
| `backend`            | str      | "torch"     | Select DL framework for inference .*["torch", "onnx"]*     |
| `type`               | str      | "mobile"    | Select pretrained-model type. *["mobile", "server"]*     |
| `conf_location`      | str      | None      | Specify config directory path! Default: None, implies use of [pre-set config](https://github.com/DocYard-ai/UCR/tree/develop/conf) downloaded from web and stored in "~/.ucr" folder.    |
| `force_download`&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;     | bool     | False     | Force download config files and pretrained models from web. Needed in case of incomplete/corrupt downloads.     |
| `output_format`      | str      | "df"      | Specify return type and structure for OCR output. *["ppocr", "df"]* Details [here](coming_soon.md)!     |
| `det`                | bool     | True      | Whether to perform Detection or not on the input data!     |
| `rec`                | bool     | True      | Whether to perform Recognition or not on the input data!     |
| `cls`                | bool     | False     | Whether to perform Classification or not on the input data!     |
| `system_overrides`&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;   | str/list | None      | Overrides arguments in [infer_system](https://github.com/DocYard-ai/UCR/blob/develop/conf/infer_system.yaml) config file. Details [here](coming_soon.md)!    |
| <span style="font-weight:bold; font-size: 125%"> Detection args|           |           |      |
| `det_algorithm`      | str      | "CRAFT"     | Specify Detection algorithm to select respective pretrained model. *["DB", "CRAFT"]* Details [here](coming_soon.md)!    |
| `det_config_name`    | str      | "infer_det" | Specify det config filename located inside `conf_location` as shown [here](https://github.com/DocYard-ai/UCR/tree/develop/conf)! More on config structure [here](coming_soon.md)!  
| `det_model_location`&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; | str      | None      | Overrides model path present in `det_config_name`.yaml file! Useful for custom trained models.      |
| `det_batch_size`     | int      | 1         |  Batch size for performing Detection on `input` data.    |
| `det_overrides`      | str/list | None      |  Overrides arguments in [infer_det](https://github.com/DocYard-ai/UCR/blob/develop/conf/infer_det.yaml) config file. Details [here](coming_soon.md)!     |
| <span style="font-weight:bold; font-size: 125%"> Recognition args|          |           |      |
| `rec_algorithm`      | str      | "CRNN"      |  Specify recognition algorithm to select respective pretrained model. *["CRNN"]* Details [here](coming_soon.md)!   |
| `rec_config_name`    | str      | "infer_rec" | Specify rec config filename located inside `conf_location` as shown [here](https://github.com/DocYard-ai/UCR/tree/develop/conf)! More on config structure [here](coming_soon.md)!   |
| `rec_model_location` | str      | None      |  Overrides model path present in `rec_config_name`.yaml file! Useful for custom trained models.   |
| `rec_batch_size`     | int      | 8         |  Batch size for performing Recognition on `input` data.    |
| `rec_overrides`      | str/list | None      |  Overrides arguments in [infer_rec](https://github.com/DocYard-ai/UCR/blob/develop/conf/infer_rec.yaml) config file. Details [here](coming_soon.md)!     |
| `rec_whitelist`      | str      | None      |  Only whitelisted characters will be considered during prediction. See example [here](coming_soon.md)! |     |
| `rec_blacklist`      | str      | None      |  Blacklisted characters will be ignored during prediction. See example [here](coming_soon.md)!   |
| <span style="font-weight:bold; font-size: 125%"> Classification args|          |           |      |
| `cls_algorithm`      | str      | "CLS"       |  Specify classification algorithm to select respective pretrained model. *["CLS"]* Details [here](coming_soon.md)!   |
| `cls_config_name`    | str      | "infer_cls" | Specify cls config filename located inside `conf_location` as shown [here](https://github.com/DocYard-ai/UCR/tree/develop/conf)! More on config structure [here](coming_soon.md)!    |
| `cls_model_location` | str      | None      |  Overrides model path present in `cls_config_name`.yaml file! Useful for custom trained models.    |
| `cls_batch_size`     | int      | 8         |   Batch size for performing Classification on `input` data.   |
| `cls_overrides`      | str/list | None      |   Overrides arguments in [infer_cls](https://github.com/DocYard-ai/UCR/blob/develop/conf/infer_cls.yaml) config file. Details [here](coming_soon.md)!    |

<!-- 
,type,default,help
conf_location,str,None,
force_download,bool,False,
i/input,str,,
o/output,str,None,
l/lang,str,"ch_sim",
d/device,str,"cuda",
backend,str,"torch",
type,str,"mobile",
output_format,str,None,
system_overrides,str/list,None,
det,bool,True,
rec,bool,True,
cls,bool,False,
verbose,bool,False,
det_algorithm,str,"CRAFT",
det_config_name,str,"infer_det",
det_model_location,str,None,
det_batch_size,int,1,
det_overrides,str/list,None,
rec_algorithm,str,"CRNN",
rec_config_name,str,"infer_rec",
rec_model_location,str,None,
rec_batch_size,int,8,
rec_overrides,str/list,None,
rec_whitelist,str,None,
rec_blacklist,str,None,
cls_algorithm,str,"CLS",
cls_config_name,str,"infer_cls",
cls_model_location,str,None,
cls_batch_size,int,8,
cls_overrides,str/list,None, 

Can be either <span style="color:#FF8856">file-path</span> (str), <span style="color:#FF8856">folder-path</span> (str), <span style="color:#FF8856">web-address</span> (str), <span style="color:#FF8856">numpy array</span> (format:[H,W,C], C:BGR/GRAY) or <span style="color:#FF8856">list</span> of str/arrays.
-->
