#-*-coding:utf-8-*-
# date:2021-03-09
# Author: Eric.Lee
# function: yolo v3 hand detect

import os
import cv2
import numpy as np
import time

import torch

from hand_detect.yolov3 import Yolov3, Yolov3Tiny
from hand_detect.utils.torch_utils import select_device
from hand_detect.acc_model import acc_model

import torch.backends.cudnn as cudnn
import torch.nn.functional as F


import random

def show_model_param(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        print("该层的结构: {}, 参数和: {}".format(str(list(i.size())), str(l)))
        k = k + l
    print("----------------------")
    print("总参数数量和: " + str(k))

def process_data(img, img_size=416):# 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RG25
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [255, 55,90], thickness=tf, lineType=cv2.LINE_AA)

def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou

def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img_size, coords, img0_shape):# image size 转为 原图尺寸
    # Rescale x1, y1, x2, y2 from 416 to image size
    # print('coords     : ',coords)
    # print('img0_shape : ',img0_shape)
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    # print('gain       : ',gain)
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    # print('pad_xpad_y : ',pad_x,pad_y)
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)# 夹紧区间最小值不为负数
    return coords

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Filter out confidence scores below threshold
        class_conf, class_pred = pred[:, 5:].max(1)  # max class_conf, index
        pred[:, 4] *= class_conf  # finall conf = obj_conf * class_conf

        i = (pred[:, 4] > conf_thres) & (pred[:, 2] > min_wh) & (pred[:, 3] > min_wh)
        # s2=time.time()
        pred2 = pred[i]
        # print("++++++pred2 = pred[i]",time.time()-s2, pred2)

        # If none are remaining => process next image
        if len(pred2) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred2[:, :4] = xywh2xyxy(pred2[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred2 = torch.cat((pred2[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        pred2 = pred2[(-pred2[:, 4]).argsort()]

        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in pred2[:, -1].unique():
            dc = pred2[pred2[:, -1] == c]  # select class c
            dc = dc[:min(len(dc), 100)]  # limit to first 100 boxes

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort
    return output

def letterbox(img, height=416, augment=False, color=(127.5, 127.5, 127.5)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    # resize img
    if augment:
        interpolation = np.random.choice([None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        if interpolation is None:
            img = cv2.resize(img, new_shape)
        else:
            img = cv2.resize(img, new_shape, interpolation=interpolation)
    else:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
    # print("resize time:",time.time()-s1)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh
#---------------------------------------------------------
# model_path = './coco_model/yolov3_coco.pt' # 检测模型路径
# root_path = './test_images/'# 测试文件夹
# model_arch = 'yolov3' # 模型类型
# voc_config = 'cfg/voc.data' # 模型相关配置文件
# img_size = 416 # 图像尺寸
# conf_thres = 0.35# 检测置信度
# nms_thres = 0.5 # nms 阈值
class yolo_v3_hand_model(object):
    def __init__(self,
        model_path = '/weights/hand_416-2021-02-20.pt',
        model_arch = 'yolov3',
        yolo_anchor_scale = 1.,
        img_size=416,
        conf_thres=0.16,
        nms_thres=0.4,):
        print("yolo v3 hand_model loading :  {}".format(model_path))
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.img_size = img_size
        self.classes = ["Hand"]
        self.num_classes = len(self.classes)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        #-----------------------------------------------------------------------
        weights = model_path
        if "tiny" in model_arch:
            a_scalse = 416./img_size*yolo_anchor_scale
            anchors=[(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]
            anchors_new = [ (int(anchors[j][0]/a_scalse),int(anchors[j][1]/a_scalse)) for j in range(len(anchors)) ]

            model = Yolov3Tiny(self.num_classes,anchors = anchors_new)
        else:
            a_scalse = 416./img_size
            anchors=[(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)]
            anchors_new = [ (int(anchors[j][0]/a_scalse),int(anchors[j][1]/a_scalse)) for j in range(len(anchors)) ]
            model = Yolov3(self.num_classes,anchors = anchors_new)
        #-----------------------------------------------------------------------

        self.model = model
        # show_model_param(self.model)# 显示模型参数

        # print('num_classes : ',self.num_classes)

        self.device = select_device() # 运行硬件选择
        self.use_cuda = torch.cuda.is_available()
        # Load weights
        if os.access(weights,os.F_OK):# 判断模型文件是否存在
            self.model.load_state_dict(torch.load(weights, map_location=lambda storage, loc: storage)['model'])
        else:
            print('------- >>> error : model not exists')
            return False
        #
        self.model.eval()#模型设置为 eval
        acc_model('',self.model)
        self.model = self.model.to(self.device)

    def predict(self, img_,vis):
        with torch.no_grad():
            t = time.time()
            img = process_data(img_, self.img_size)
            t1 = time.time()
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)

            pred, _ = self.model(img)#图片检测

            t2 = time.time()
            detections = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0] # nms
            t3 = time.time()
            # print("t3 time:", t3)

            if (detections is None) or len(detections) == 0:
                return []
            # Rescale boxes from 416 to true image size
            detections[:, :4] = scale_coords(self.img_size, detections[:, :4], img_.shape).round()
            # 绘制检测结果 ：detect reslut
            dets_for_landmarks = []
            colors = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, 10 + 1)][::-1]

            output_dict_ = []
            for *xyxy, conf, cls_conf, cls in detections:
                label = '%s %.2f' % (self.classes[0], conf)
                x1,y1,x2,y2 = xyxy
                output_dict_.append((float(x1),float(y1),float(x2),float(y2),float(conf.item())))
                if vis:
                    plot_one_box(xyxy, img_, label=label, color=(0,175,255), line_thickness = 2)
            return output_dict_
