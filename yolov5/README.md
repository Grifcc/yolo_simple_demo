# Cambricon Guide

## run

### parameter and typical command

    * parameter1: int8 mode, int16 mode, channel_int8 mode, channel_int16 mode
    * parameter2: device select MLU270 or MLU220

```sh
    ./run_all_offline_mc.sh [1|2|3|4] [1|2]
    eg. ./run_all_offline_mc.sh 1 1
        which means running int8 offline model on MLU270.
```
## Limitations(offline)

**option: core number[1,4,16], batch size [1, 4, 8, 16], quantization_mode(1:int8, 2:int16, 3:channel_int8, 4:channel_int16),  data_type(0:float32, 1:half))**

- **MLU270**

maximum support:
|core_number|batch_size|quantized_mode|data_type|
|:---|:--|:--|:--|
|1/4/16|1/4/8/16|int8/int16|float16/float32|

- **MLU220 M.2**

|core_number|batch_size|quantized_mode|data_type|
|:---|:--|:--|:--|
|1/4|1/4/8|int8/int16|float16/float32|


- **MLU220 Edge**

|core_number|batch_size|quantized_mode|data_type|
|:---|:--|:--|:--|
|1/4|1/4|int8/int16|float16/float32|


## environment

 ` TORCH_HOME=/path/to/torch/model` <br>
 ` COCO_PATH_PYTORCH=/path/to/coco/dataset/`

