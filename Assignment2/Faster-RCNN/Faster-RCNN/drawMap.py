import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
import glob

from frcnn import FRCNN
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

classes_path    = 'model_data/voc_classes.txt'
MINOVERLAP      = 0.5
map_vis         = False
VOCdevkit_path  = 'VOCdevkit'
map_out_path    = 'map_curve_out'
log_dir    = os.path.join('./logs', 'mAPs')
writer     = SummaryWriter(log_dir)

image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/val.txt")).read().strip().split()

if not os.path.exists(map_out_path):
    os.makedirs(map_out_path)
    os.makedirs(os.path.join(map_out_path, 'detection-results'))
    os.makedirs(os.path.join(map_out_path, 'ground-truth'))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

class_names, _ = get_classes(classes_path)

print("Get ground truth result.")
for image_id in tqdm(image_ids):
    with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
        root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
        for obj in root.findall('object'):
            difficult_flag = False
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                if int(difficult)==1:
                    difficult_flag = True
            obj_name = obj.find('name').text
            if obj_name not in class_names:
                continue
            bndbox  = obj.find('bndbox')
            left    = bndbox.find('xmin').text
            top     = bndbox.find('ymin').text
            right   = bndbox.find('xmax').text
            bottom  = bndbox.find('ymax').text

            if difficult_flag:
                new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
            else:
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
print("Get ground truth result done.")

model_list = glob.glob('./logs' + '/*.pth')
model_list.sort()
maps = []
iters = []
for (i, model_path) in enumerate(model_list):
    epoch = (i + 1) * 5
    print("Load model.")
    frcnn = FRCNN(model_path = model_path, confidence = 0.01, nms_iou = 0.5)
    print("Load model done.")

    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
        image       = Image.open(image_path)
        frcnn.get_map_txt(image_id, image, class_names, map_out_path)
    print("Get predict result done.")
    

    print("Get map.")
    mAP = get_map(MINOVERLAP, True, path = map_out_path)
    print("Get map done.")

    maps.append(mAP)

    with open(os.path.join(log_dir, "epoch_map.txt"), 'a') as f:
        f.write(str(mAP))
        f.write("\n")

    writer.add_scalar('mAP', mAP, epoch)

    iters.append(epoch)

    plt.figure()
    plt.plot(iters, maps, 'blue', linewidth = 2, label='val mAP')
    try:
        if len(maps) < 25:
            num = 5
        else:
            num = 15
        
        plt.plot(iters, scipy.signal.savgol_filter(maps, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth val mAP')
    except:
        pass

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(log_dir, "epoch_map.png"))

    plt.cla()
    plt.close("all")
