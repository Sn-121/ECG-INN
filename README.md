# Edge and Color Guided Invertible Neural Network for Retinex-based Deep Image Enhancement

### 1. Datasets

Please download these three low-light enhancement datasets(LOL,HUAWEI,LOLv2)

### 2. Pretrained Models

Download pretrained model weights from "(https://drive.google.com/drive/folders/1kkgJFc-syFLPQS0lueNtiN2ibeBxZZxv?usp=drive_link)" and put them into folder './pretrain/'.

### 3. Test Models
Change the weight path for decom network in "/models/condition/condition_retinex.py"
Change the weight path for color and edge network in "/models/archs/Enhance_arch.py"
Change the weight path for main network(INN) in "/options/test_Enhance_LOL.yml" ， it also contains paths to the images needed for testing.
Run 'eval.py' for testing.

### 4.Train Models
Change the path of "create_txt.py" and run it to create the list of datasets.
Change the weight path for main network(INN) in "/options/train_Enhance_LOL.yml"
Run 'train.py' for training or run 'eval.py' for testing.

```
python train.py
```
```
python eval.py
```

