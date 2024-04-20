# Edge and Color Guided Invertible Neural Network for Retinex-based Deep Image Enhancement
## Recommended Environment

python  3.7.4
pytorch  1.9.1+cu111
torchvision  0.10.1+cu111
numpy  1.18.1
timm  0.6.12

### 1. Datasets

Please download these three low-light enhancement datasets :  
LOL [https://daooshee.github.io/BMVC2018website/],  
HUAWEI [https://github.com/JianghaiSCU/R2RNet],  
LOLv2 [https://github.com/flyywh/CVPR-2020-Semi-Low-Light].

### 2. Pretrained Models

Download pretrained model weights from [https://drive.google.com/drive/folders/1kkgJFc-syFLPQS0lueNtiN2ibeBxZZxv?usp=drive_link] and put them into folder './pretrain/'.

### 3. Test Models
Change the weight path for decom network in "/models/condition/condition_retinex.py",  
Change the weight path for color and edge network in "/models/archs/Enhance_arch.py",  
Change the weight path for main network(INN) in "/options/test_Enhance_LOL.yml"，it also contains paths to the images needed for testing.  
At last, run "eval.py" for testing.

```
python eval.py
```
