!!! important "Important Notice"
    This is a **Quick Start** guide on how to use the UCR library with its existing pretrained models. For detailed information regarding its usage and advanced functionality, please refer to other sections!

## Setup

### Installation

1. <span style="color:#FF8856">Prerequisites:</span> Install compatible version of Pytorch and torchvision from [official repository](https://pytorch.org/get-started/locally/).
2. <span style="color:#FF8856">Installation:</span> Install the latest stable version of UCR:
```bash
pip install -U ucr
```

### Test installation

<span style="color:#FF8856">Test Installation (Optional):</span> Run dummy tests!
```bash
ucr test 
# Optional: Add -l/--lang='language_id' to test on particular language! 
ucr test -l='en_number'
```  

*For other installation modes and complete setup instructions, see [here](coming_soon.md)!*

## Run Prediction

1. <span style="color:#FF8856">Using Command Line:</span>
```bash
# run prediction on input file/folder
ucr predict input_path -o=output_path -l=language_id
```
List of all supported languages with their corresponding ids is shown [here](modelzoo.md)!  
To view all available options for the CLI application:
```bash
# view all CLI predict commands
ucr predict --help
```
**Usage:**   ucr predict **"input_path"** *--arguments* <span style="color:#FF8856">[ARGS]</span>  
**Returns:** Python Dictionary of either Dataframes(default) or Lists(if `--return_type="list"`).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Key:** <u>*filepath*</u> ; **Value:** <u>*df/list*</u> 
1. <span style="color:#FF8856">From Python scripts/Jupyter Notebooks:</span>
```python
# Import and Initialize
from ucr import UCR
ocr = UCR(l='en_number') # use l or lang='language_id'.

# Run Predictions
result = ocr.predict('input_path', o='output_path')
# Returns dictionary of filepath:dataframe by default.
# Set return_type="list" for filepath:list of lists. 
```  

*Refer to the [Demo](demo.md) page and/or [Colab Notebook](coming_soon.md) to see it in action!*

## Argument List 

|    <span style="font-weight:bold; font-size: 125%">Name</span>                  | <span style="font-weight:bold; font-size: 125%">Type</span>     | <span style="font-weight:bold; font-size: 125%">Default</span>   | <span style="font-weight:bold; font-size: 125%">Help</span> |
|----------------------|----------|-----------|------|
| `input`            | str/array/list      | <span style="font-style: italic; color:#FF8856">Required</span>          |Path to input location, eg: file/folder path, web address, numpy array etc. More details [here](coming_soon.md)!|
| `o/output`           | str      | "output"      | Path to output folder. <span style="font-style: italic; color:#FF8856">(optional)</span> If both `save_tsv` and `save_image` are false, no output will be saved |
| `l/lang`             | str      | "ch_sim"    | List of supported language-ids can be found [here](coming_soon.md)!     |
| `d/device`           | str      | "cuda"      | Specify device to run on. *["cuda", "cpu"]*      |
| `return_type`      | str      | "df"      | Specify return type and structure for OCR output. *["list", "df"]* Details [here](coming_soon.md)!     |
| `save_tsv`            | bool     | True     | If True, saves tab separated files inside `tsv` directory in `output` folder      |  
| `save_image`            | bool     | False     | If True, saves image inside `image` directory in `output` folder     |  
| `verbose`            | bool     | False     | Enable it to print info on console!     |  
| `backend`            | str      | "torch"     | Select DL framework for inference .*["torch", "onnx"]*     |
| `type`               | str      | "mobile"    | Select pretrained-model type. *["mobile", "server"]*     |
| `conf_location`      | str      | None      | Specify config directory path! Default: None, implies use of [pre-set config](https://github.com/DocYard-ai/UCR/tree/develop/conf) downloaded from web and stored in "~/.ucr" folder.    |
| `force_download`&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;     | bool     | False     | Force download config files and pretrained models from web. Needed in case of incomplete/corrupt downloads.     |
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
| `rec_whitelist`      | str      | None      |  Only whitelisted characters will be considered during prediction. See example [here](demo.md)! |     |
| `rec_blacklist`      | str      | None      |  Blacklisted characters will be ignored during prediction. See example [here](demo.md)!   |
| <span style="font-weight:bold; font-size: 125%"> Classification args|          |           |      |
| `cls_algorithm`      | str      | "CLS"       |  Specify classification algorithm to select respective pretrained model. *["CLS"]* Details [here](coming_soon.md)!   |
| `cls_config_name`    | str      | "infer_cls" | Specify cls config filename located inside `conf_location` as shown [here](https://github.com/DocYard-ai/UCR/tree/develop/conf)! More on config structure [here](coming_soon.md)!    |
| `cls_model_location` | str      | None      |  Overrides model path present in `cls_config_name`.yaml file! Useful for custom trained models.    |
| `cls_batch_size`     | int      | 8         |   Batch size for performing Classification on `input` data.   |
| `cls_overrides`      | str/list | None      |   Overrides arguments in [infer_cls](https://github.com/DocYard-ai/UCR/blob/develop/conf/infer_cls.yaml) config file. Details [here](coming_soon.md)!    |
