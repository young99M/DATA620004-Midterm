# **Assignment 1**

Using ShuffleNet for CIFAR100 image classification task. 

### **Requirements**
This is my experiment environment:
- python 3.7.11
- pytorch 1.10.2+cu113

### **Usage**
#### **1. enter directory**
``` bash
cd Assignment1
```

#### **2. run tensorboard(optional)**
Install tensorboard
```bash
pip install tensorboard
mkdir runs
tensorboard --logdir='runs' --port=6006 -- host='localhost'
```
the log file of  tensorboard can be found at `./Assignment1/runs`.

#### **3. train the model**
You need to specify the net you want to train using arg `-net`
```bash
# train shufflenet
python train.py -net shufflenet
python train_cutout.py -net shufflenet
python train_mixup.py -net shufflenet
python train_cutmix.py -net shufflenet
```
If you want to use warmup training for more epochs, set `-warm` to the number you want.

#### **4. test the model**
Test the model using `test.py`
```bash
python test.py -net shufflenet -type baseline
```
Test other data augmentation methods by setting `-type` to cutout, mixup or cutmix.



# **Assignment 2**
## **Faster-RCNN**
This is a pytorch implementation of Faster R-CNN following the work of https://github.com/bubbliiiing/faster-rcnn-pytorch with some adaptations. The dataset used is PASCAL VOC 2007.

### **Requirements**
My experiment environment:
>torch == 1.11.0 CUDA Version: 11.3

### **Training**
1. Using VOC format for training, you need to download the VOC07 data set, decompress it and put it in the root directory.
2. Modify `annotation_mode = 2` in `voc_annotation.py`, then
```bash
python voc_annotation.py
```
generate `2007_train.txt` and `2007_val.txt` in the root directory.

3. The default parameters of `train.py` are used to the VOC dataset, and start training by
```bash
python train.py
```
If you set the pretrained backbone network (ResNet50 by default), it will automatically download `resnet50-19c8e357.pth` under the `model_data` path. The plotted curves and the trained model parameters will be stored in the logs folder.

### **Predict**
1.Prediction requires two files, `frcnn.py` and `predict.py`. Modify the parameter `model_path` when initializing the model in `predict.py` to refer to the trained weights file.

2. 
```bash
python predcit.py
```
After running, enter the image path to detect.

### **Evaluation**
1. Modify the parameter *model_path* when *get_map.py* initializes the model to refer to the trained weights file.
 
2. To get the evaluation results (saved in the *map_out* folder), run
```bash
python get_map.py
```

## **YOLOv3**
### **Introduction**
This is a pytorch implementation of YOLOv3 following the work of [Peterisfar](https://github.com/Peterisfar/YOLOV3)  with some adaptations. The dataset used is PASCAL VOC 2007. The trained YOLOv3 model could reach an mAP  $75.8\%$ on VOC test 2007.


---
### **Environment**

```bash
pip3 install -r requirements.txt --user
```

---
### **Prepared work**

#### 1、Download dataset
* Download Pascal VOC dataset : [VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)、[VOC2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar). put them in the dir, and update the `"DATA_PATH"` in the params.py.
* Convert data format : 

```bash
cd YOLOV3 && mkdir data
cd utils
python3 voc.py # get train_annotation.txt and test_annotation.txt in data/
```

#### 2、Download weight file
Darknet pre-trained weight :  [darknet53-448.weights](https://pjreddie.com/media/files/darknet53_448.weights) 

Make dir `weight/` in the YOLOV3 and put the weight file in.

---
### **Train**
Run the following command to start training and see the details in the `config/yolov3_config_voc.py`

```Bash
WEIGHT_PATH=weight/darknet53_448.weights

CUDA_VISIBLE_DEVICES=0 nohup python3 -u train.py --weight_path $WEIGHT_PATH --gpu_id 0 > nohup.log 2>&1 &
```

---
### **Test**
Define the weight file path `WEIGHT_FILE` and test data's path `DATA_TEST`
```Bash
WEIGHT_PATH=weight/best.pt
DATA_TEST=./data/test # tets images

CUDA_VISIBLE_DEVICES=0 python3 test.py --weight_path $WEIGHT_PATH --gpu_id 0 --visiual $DATA_TEST --eval
```
---
### Plots

I use tensorboard to plot the loss curve and mAP curve, the log file of  tensorboard can be found at `./YOLOV3/runs`. One may just open tensorboard with  `cd YOLOV3` and see the plots.
