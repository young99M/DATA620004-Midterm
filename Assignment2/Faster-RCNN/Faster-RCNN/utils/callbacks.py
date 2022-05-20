import datetime
import os
# from tqdm import tqdm
# import xml.etree.ElementTree as ET
# from utils.utils import get_classes
# from frcnn import FRCNN
# from PIL import Image

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.losses     = []
        self.val_loss   = []
        # self.maps = []
        # self.map_out_path = 'train_map_out'
        # self.GT_PATH = os.path.join(self.map_out_path, 'ground-truth')
        # self.DR_PATH = os.path.join(self.map_out_path, 'detection-results')
        # self.VOCdevkit_path  = 'VOCdevkit'
        # self.image_ids = open(os.path.join(self.VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
        # self.class_names, _ = get_classes('model_data/voc_classes.txt')
        # print('aaa', self.class_names)
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        # try:
        #     dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummy_input)
        # except:
        #     pass
        # self.makeGT(self.map_out_path)


    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    # def makeGT(self, map_out_path):
    #     if not os.path.exists(map_out_path):
    #         os.makedirs(map_out_path)
    #     if not os.path.exists(self.GT_PATH):
    #         os.makedirs(self.GT_PATH)
    #         print('here3')
    #         print("Get ground truth result.")
    #         for image_id in tqdm(self.image_ids):
    #             with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
    #                 root = ET.parse(os.path.join(self.VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
    #                 for obj in root.findall('object'):
    #                     difficult_flag = False
    #                     if obj.find('difficult')!=None:
    #                         difficult = obj.find('difficult').text
    #                         if int(difficult)==1:
    #                             difficult_flag = True
    #                     obj_name = obj.find('name').text
    #                     if obj_name not in self.class_names:
    #                         continue
    #                     bndbox  = obj.find('bndbox')
    #                     left    = bndbox.find('xmin').text
    #                     top     = bndbox.find('ymin').text
    #                     right   = bndbox.find('xmax').text
    #                     bottom  = bndbox.find('ymax').text

    #                     if difficult_flag:
    #                         new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
    #                     else:
    #                         new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    #         print("Get ground truth result done.")

    # def getDetectionResults(self, model_path):
    #     # os.makedirs(self.DR_PATH)
    #     print("Load model.")
    #     frcnn = FRCNN(model_path=model_path, confidence = 0.01, nms_iou = 0.5)
    #     print("Load model done.")

    #     print("Get predict result.")
    #     for image_id in tqdm(self.image_ids):
    #         image_path  = os.path.join(self.VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
    #         image       = Image.open(image_path)
    #         frcnn.get_map_txt(image_id, image, self.class_names, self.map_out_path)
    #     print("Get predict result done.")

    # def voc_ap(rec, prec):
    #     """
    #     --- Official matlab code VOC2012---
    #     mrec=[0 ; rec ; 1];
    #     mpre=[0 ; prec ; 0];
    #     for i=numel(mpre)-1:-1:1
    #             mpre(i)=max(mpre(i),mpre(i+1));
    #     end
    #     i=find(mrec(2:end)~=mrec(1:end-1))+1;
    #     ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    #     """
    #     rec.insert(0, 0.0) # insert 0.0 at begining of list
    #     rec.append(1.0) # insert 1.0 at end of list
    #     mrec = rec[:]
    #     prec.insert(0, 0.0) # insert 0.0 at begining of list
    #     prec.append(0.0) # insert 0.0 at end of list
    #     mpre = prec[:]
    #     """
    #     This part makes the precision monotonically decreasing
    #         (goes from the end to the beginning)
    #         matlab: for i=numel(mpre)-1:-1:1
    #                     mpre(i)=max(mpre(i),mpre(i+1));
    #     """
    #     for i in range(len(mpre)-2, -1, -1):
    #         mpre[i] = max(mpre[i], mpre[i+1])
    #     """
    #     This part creates a list of indexes where the recall changes
    #         matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    #     """
    #     i_list = []
    #     for i in range(1, len(mrec)):
    #         if mrec[i] != mrec[i-1]:
    #             i_list.append(i) # if it was matlab would be i + 1
    #     """
    #     The Average Precision (AP) is the area under the curve
    #         (numerical integration)
    #         matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    #     """
    #     ap = 0.0
    #     for i in i_list:
    #         ap += ((mrec[i]-mrec[i-1])*mpre[i])
    #     return ap, mrec, mpre
        
    # def error(msg):
    #     print(msg)
    #     sys.exit(0)

    # def file_lines_to_list(path):
    #     # open txt file lines to a list
    #     with open(path) as f:
    #         content = f.readlines()
    #     # remove whitespace characters like `\n` at the end of each line
    #     content = [x.strip() for x in content]
    #     return content

    # def update_map(self, epoch, model_path, MINOVERLAP):
    #     self.getDetectionResults(model_path)
    #     TEMP_FILES_PATH     = os.path.join(self.map_out_path, '.temp_files')
    #     if not os.path.exists(TEMP_FILES_PATH):
    #         os.makedirs(TEMP_FILES_PATH)
    #     ground_truth_files_list = glob.glob(self.GT_PATH + '/*.txt')
    #     if len(ground_truth_files_list) == 0:
    #         error("Error: No ground-truth files found!")
    #     ground_truth_files_list.sort()
    #     gt_counter_per_class     = {}
    #     counter_images_per_class = {}

    #     for txt_file in ground_truth_files_list:
    #         file_id     = txt_file.split(".txt", 1)[0]
    #         file_id     = os.path.basename(os.path.normpath(file_id))
    #         temp_path   = os.path.join(self.DR_PATH, (file_id + ".txt"))
    #         if not os.path.exists(temp_path):
    #             error_msg = "Error. File not found: {}\n".format(temp_path)
    #             error(error_msg)
    #         lines_list      = file_lines_to_list(txt_file)
    #         bounding_boxes  = []
    #         is_difficult    = False
    #         already_seen_classes = []
    #         for line in lines_list:
    #             try:
    #                 if "difficult" in line:
    #                     class_name, left, top, right, bottom, _difficult = line.split()
    #                     is_difficult = True
    #                 else:
    #                     class_name, left, top, right, bottom = line.split()
    #             except:
    #                 if "difficult" in line:
    #                     line_split  = line.split()
    #                     _difficult  = line_split[-1]
    #                     bottom      = line_split[-2]
    #                     right       = line_split[-3]
    #                     top         = line_split[-4]
    #                     left        = line_split[-5]
    #                     class_name  = ""
    #                     for name in line_split[:-5]:
    #                         class_name += name + " "
    #                     class_name  = class_name[:-1]
    #                     is_difficult = True
    #                 else:
    #                     line_split  = line.split()
    #                     bottom      = line_split[-1]
    #                     right       = line_split[-2]
    #                     top         = line_split[-3]
    #                     left        = line_split[-4]
    #                     class_name  = ""
    #                     for name in line_split[:-4]:
    #                         class_name += name + " "
    #                     class_name = class_name[:-1]

    #             bbox = left + " " + top + " " + right + " " + bottom
    #             if is_difficult:
    #                 bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
    #                 is_difficult = False
    #             else:
    #                 bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
    #                 if class_name in gt_counter_per_class:
    #                     gt_counter_per_class[class_name] += 1
    #                 else:
    #                     gt_counter_per_class[class_name] = 1

    #                 if class_name not in already_seen_classes:
    #                     if class_name in counter_images_per_class:
    #                         counter_images_per_class[class_name] += 1
    #                     else:
    #                         counter_images_per_class[class_name] = 1
    #                     already_seen_classes.append(class_name)

    #         with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
    #             json.dump(bounding_boxes, outfile)

    #     gt_classes  = list(gt_counter_per_class.keys())
    #     gt_classes  = sorted(gt_classes)
    #     n_classes   = len(gt_classes)

    #     dr_files_list = glob.glob(self.DR_PATH + '/*.txt')
    #     dr_files_list.sort()
    #     for class_index, class_name in enumerate(gt_classes):
    #         bounding_boxes = []
    #         for txt_file in dr_files_list:
    #             file_id = txt_file.split(".txt",1)[0]
    #             file_id = os.path.basename(os.path.normpath(file_id))
    #             temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
    #             if class_index == 0:
    #                 if not os.path.exists(temp_path):
    #                     error_msg = "Error. File not found: {}\n".format(temp_path)
    #                     error(error_msg)
    #             lines = file_lines_to_list(txt_file)
    #             for line in lines:
    #                 try:
    #                     tmp_class_name, confidence, left, top, right, bottom = line.split()
    #                 except:
    #                     line_split      = line.split()
    #                     bottom          = line_split[-1]
    #                     right           = line_split[-2]
    #                     top             = line_split[-3]
    #                     left            = line_split[-4]
    #                     confidence      = line_split[-5]
    #                     tmp_class_name  = ""
    #                     for name in line_split[:-5]:
    #                         tmp_class_name += name + " "
    #                     tmp_class_name  = tmp_class_name[:-1]

    #                 if tmp_class_name == class_name:
    #                     bbox = left + " " + top + " " + right + " " +bottom
    #                     bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})

    #         bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
    #         with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
    #             json.dump(bounding_boxes, outfile)

    #     sum_AP = 0.0

    #     for class_index, class_name in enumerate(gt_classes):
    #         count_true_positives[class_name] = 0
    #         dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
    #         dr_data = json.load(open(dr_file))

    #         nd          = len(dr_data)
    #         tp          = [0] * nd
    #         fp          = [0] * nd
    #         score       = [0] * nd
    #         score05_idx = 0
    #         for idx, detection in enumerate(dr_data):
    #             file_id     = detection["file_id"]
    #             score[idx]  = float(detection["confidence"])
    #             if score[idx] > 0.5:
    #                 score05_idx = idx

    #             gt_file             = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
    #             ground_truth_data   = json.load(open(gt_file))
    #             ovmax       = -1
    #             gt_match    = -1
    #             bb          = [float(x) for x in detection["bbox"].split()]
    #             for obj in ground_truth_data:
    #                 if obj["class_name"] == class_name:
    #                     bbgt    = [ float(x) for x in obj["bbox"].split() ]
    #                     bi      = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
    #                     iw      = bi[2] - bi[0] + 1
    #                     ih      = bi[3] - bi[1] + 1
    #                     if iw > 0 and ih > 0:
    #                         ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
    #                                         + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
    #                         ov = iw * ih / ua
    #                         if ov > ovmax:
    #                             ovmax = ov
    #                             gt_match = obj

                        
    #             min_overlap = MINOVERLAP
    #             if ovmax >= min_overlap:
    #                 if "difficult" not in gt_match:
    #                     if not bool(gt_match["used"]):
    #                         tp[idx] = 1
    #                         gt_match["used"] = True
    #                         count_true_positives[class_name] += 1
    #                     else:
    #                         fp[idx] = 1
    #             else:
    #                 fp[idx] = 1


    #         cumsum = 0
    #         for idx, val in enumerate(fp):
    #             fp[idx] += cumsum
    #             cumsum += val
                
    #         cumsum = 0
    #         for idx, val in enumerate(tp):
    #             tp[idx] += cumsum
    #             cumsum += val

    #         rec = tp[:]
    #         for idx, val in enumerate(tp):
    #             rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)

    #         prec = tp[:]
    #         for idx, val in enumerate(tp):
    #             prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)

    #         ap, mrec, mprec = voc_ap(rec[:], prec[:])

    #         sum_AP  += ap

    #     mAP     = sum_AP / n_classes
    #     text    = "mAP = {0:.2f}%".format(mAP*100)
    #     results_file.write(text + "\n")
    #     print(text)
    #     shutil.rmtree(TEMP_FILES_PATH)

    #     self.maps.append(mAP)

    #     with open(os.path.join(self.log_dir, "epoch_mAP.txt"), 'a') as f:
    #         f.write(str(mAP))
    #         f.write("\n")

    #     self.writer.add_scalar('loss', mAP, epoch)
    #     self.mAP_plot()

    # def mAP_plot(self):
    #     iters = range(len(self.maps))

    #     plt.figure()
    #     plt.plot(iters, self.maps, 'blue', linewidth = 2, label='val mAP')
    #     try:
    #         if len(self.maps) < 25:
    #             num = 5
    #         else:
    #             num = 15
            
    #         plt.plot(iters, scipy.signal.savgol_filter(self.maps, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth val mAP')
    #     except:
    #         pass

    #     plt.grid(True)
    #     plt.xlabel('Epoch')
    #     plt.ylabel('mAP')
    #     plt.legend(loc="upper right")

    #     plt.savefig(os.path.join(self.log_dir, "epoch_mAP.png"))

    #     plt.cla()
    #     plt.close("all")
