#!/usr/bin/env python3

"""
    Modified from https://github.com/mikel-brostrom/yolov7/blob/main/track.py
    Updated by Jahnavi Malagavalli and Tong Wang, From Jan 2023

    Vehicle & License Plate detection script using YOLOv7, Real-ESRGAN, finetuned EasyOCR and StrongSORT

    Usage:  Follow the steps in the README file and then run the below command
    python main.py --conf-thres 0.25 --source ../test/20230222_115854.mp4 --device 0 --save-crop-lp --save-crop 
    --save-vid  --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov7.pt --classes 1 2 3 5 7 
"""

from ultralytics import YOLO
#from ultralytics.utils.ops import non_max_suppression
import inspect  

import argparse
import matplotlib.pyplot as plt
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import cv2
import torch.backends.cudnn as cudnn
from numpy import random
from lu_vp_detect import VPDetection
import copy
length_thresh = 50

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams, letterbox
from yolov7.utils.general import (check_img_size, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file, non_max_suppression)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

from extract_vp_utils import *
import easyocr

import json

#----------------------------------------for SR (Super Resolution)
import glob
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
#-----------------------------------------

#--------------------- for make and model
import pandas as pd
from yolov7.predict_mmr import mmr_predict,initialize_model,Image,transforms
from collections import defaultdict
#---------------------


#--------------------- for color detection
from color_recognition.color_recognition_api import color_histogram_feature_extraction
from color_recognition.color_recognition_api import knn_classifier
from colordetect import ColorDetect
#---------------------

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

# This function descirbes our SR model
def SR(args):
    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    
    # determine model paths
    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength (Gaussian Noise)
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.device)
    return upsampler


def get_contours(vp1, vp2, vp3, box, mask):
    if vp1[0] > vp2[0]:
        tmp = vp1
        vp1 = vp2
        vp2 = tmp

    mid_x = (box[1] + box[3]) // 2

    mask_frame = np.array(mask)
    _, mask_frame = cv2.threshold(mask_frame, 0.4, 255, cv2.THRESH_BINARY)
    mask_frame = (mask_frame * 255).astype(np.uint8)
    countours, _ = cv2.findContours(mask_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    hull = cv2.convexHull(countours[0])
    pts = hull[:, 0, :]
    
    p1, p2 = find_cornerpts(vp1, pts)
    p3, p4 = find_cornerpts(vp2, pts)
    p5, p6 = find_cornerpts(vp3, pts)
    p1 = pts[p1]
    p2 = pts[p2]
    p3 = pts[p3]
    p4 = pts[p4]
    p5 = pts[p5]
    p6 = pts[p6]

    vp1l1 = line(vp1, p1)
    # left side
    vp1l2 = line(vp1, p2)
    # right side
    vp2l1 = line(vp2, p3)
    # left side
    vp2l2 = line(vp2, p4)
    # left side
    vp3l1 = line(vp3, p5)
    # right side
    vp3l2 = line(vp3, p6)

    vp1l1_vp2l1 = intersection(vp1l1, vp2l1)
    vp1l1_vp2l2 = intersection(vp1l1, vp2l2)
    # print(vp1l1_vp2l2)
    if vp1l1_vp2l1 and ((vp1l1_vp2l1[0] - p5[0]) * (vp1l1_vp2l1[0] - p6[0])) <= 0:
        vp1l2_vp2l2 = intersection(vp1l2, vp2l2)
        if vp1l1_vp2l1[1] > vp1l2_vp2l2[1]:
            bottom_point = vp1l1_vp2l1
            used_vp1l = vp1l1
            used_vp2l = vp2l1
        else:
            bottom_point = vp1l2_vp2l2
            used_vp1l = vp1l2
            used_vp2l = vp2l2
        # inside
        fixp1_1 = intersection(used_vp1l, vp3l1)
        fixp1_2 = intersection(used_vp1l, vp3l2)
        if (abs(fixp1_1[0] - vp1[0]) < abs(fixp1_2[0] - vp1[0])):
            fixp1 = fixp1_1
            fixp2 = intersection(used_vp2l, vp3l2)
        else:
            fixp1 = fixp1_2
            fixp2 = intersection(used_vp2l, vp3l1)
        fixp3 = bottom_point
    elif vp1l1_vp2l2 and ((vp1l1_vp2l2[0] - p5[0]) * (vp1l1_vp2l2[0] - p6[0])) <= 0:
        vp1l2_vp2l1 = intersection(vp1l2, vp2l1)
        if vp1l1_vp2l2[1] > vp1l2_vp2l1[1]:
            bottom_point = vp1l1_vp2l2
            used_vp1l = vp1l1
            used_vp2l = vp2l2
        else:
            bottom_point = vp1l2_vp2l1
            used_vp1l = vp1l2
            used_vp2l = vp2l1
        # inside
        fixp1_1 = intersection(used_vp1l, vp3l1)
        fixp1_2 = intersection(used_vp1l, vp3l2)
        if (abs(fixp1_1[0] - vp1[0]) < abs(fixp1_2[0] - vp1[0])):
            fixp1 = fixp1_1
            fixp2 = intersection(used_vp2l, vp3l2)
        else:
            fixp1 = fixp1_2
            fixp2 = intersection(used_vp2l, vp3l1)
        fixp3 = bottom_point
    else:
        fixp1 = fixp2 = fixp3 = None
    
    return fixp1, fixp2, fixp3


def pretty_line(img: np.array, p1, p2, color,
                thickness: int) -> np.array:
    f2int = lambda p: (int(p[0]), int(p[1]))
    img = cv2.line(img, f2int(p1), f2int(p2), (0, 0, 0), thickness + 5, cv2.LINE_AA)
    img = cv2.line(img, f2int(p1), f2int(p2), color, thickness, cv2.LINE_AA)
    return img

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_crop_lp=False,  # save cropped LP prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        denoise_strength=0.5,
        outscale = 4,
        tile = 0,
        tile_pad = 10,
        pre_pad = 0,
        fp32 = False,
        alpha_upsampler = 'realesrgan',
        model_name = "RealESRGAN_x4plus",
        fps=24,
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
        yolo_weights = Path(yolo_weights[0])
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    save_dir = Path(save_dir)
    # make dir for storing the output
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 

    # Load model
    device = select_device(device)
    
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    #model = attempt_load(Path(yolo_weights), map_location=device)  # load FP32 model

    #New Yolo model
    model = YOLO(Path(yolo_weights))
    names = model.names
    stride = 32
    #stride = model.stride.max().cpu().numpy()  # model stride

    imgsz = check_img_size(imgsz[0], s=stride)  # check image size
    print("\n\nmodel stride:",stride)
    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources
    frame_masks = [None] * nr_sources
    previous_pos = dict()
    previous_frames = dict()
    mean_speed = defaultdict(list)

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # LP Detection initialization
    #--------------------------------------------------------------------#
    lp_weights, trace =  "weights/lp_best.pt", False
    # Load model
    lp_model = attempt_load(Path(lp_weights), map_location=device)  # load FP32 model
    stride = int(lp_model.stride.max())  # model stride
    lp_names, = lp_model.names
    #--------------------------------------------------------------------#

    #Make and model initialization
    #--------------------------------------------------------------------#
    model_name="resnet50_largeData_mmr.pt"#"resnet50_40epochs.pt_100epochs.pt"# #"resnet50_40epochs_mmr.pt"
    mmr_k=3
    #img = Image.open(path)
    #if img.mode != "RGB":  # Convert png to jpg
    #    img = img.convert("RGB")
    mmr_img_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # Finetuned MMR model, trained on 15K web scraped images
    mmr_df = pd.read_csv("yolov7/data/customData_large_ds.csv") #customData.csv") #
    mmr_num_classes = mmr_df["Classname"].nunique()
    
    mmr_df["Classencoded"] = mmr_df["Classname"].factorize()[0]

    print("model name:",model_name[:8])
    print("\n\nMMR classes:",mmr_num_classes)
    mmr_model, _ = initialize_model(model_name[:8], mmr_num_classes, feature_extract=True)
    path = os.path.dirname(__file__)
    mmr_model.load_state_dict(torch.load(os.path.join(path, "yolov7/models/" + str(model_name)), map_location=device))
    mmr_model.to(device)
    mmr_model.eval()

    # MMR with Stanford and VMMR dataset
    model_old_name="resnet50_40epochs_mmr.pt"
    mmr_old_k=3

    mmr_old_df = pd.read_pickle("yolov7/data/preprocessed_data_mmr.pkl")
    #mmr_old_df["Classname"]=mmr_old_df["Classname"].str.replace('\d{4}','', regex=True)
    mmr_old_num_classes = mmr_old_df["Classname"].nunique()
    
    mmr_old_df["Classencoded"] = mmr_old_df["Classname"].factorize()[0]

    print("model name:",model_old_name[:8])
    print("\n\nMMR classes:",mmr_old_num_classes)
    mmr_old_model, _ = initialize_model(model_old_name[:8], mmr_old_num_classes, feature_extract=True)
    path = os.path.dirname(__file__)
    mmr_old_model.load_state_dict(torch.load(os.path.join(path, "yolov7/models/" + str(model_old_name)), map_location=device))
    mmr_old_model.to(device)
    mmr_old_model.eval()
    #--------------------------------------------------------------------#

    # Load the finetuned ocr model, custom_example: name of the fine tuned pth file
    easyocr_reader = easyocr.Reader(['en'],recog_network='custom_example')

    # Loading the SR model using the arguments defined in main (see opt)
    upsampler = SR(opt)

    # Format of the output in the tracks folder for the input video
    data = {'frame_id': 0, 'vehc_id':0, 'vehc_cls':'', 'vehc_bb':[], 'vehc_conf': 0, 'vehc_color':[],'car_mmr':'','car_sf_v_mmr':'','LP_bb':[], 'LP_conf':0, 'LP_txt':'', 'LP_txt_conf':0,'vehc_speed':0}
    
    # VP detection module
    vpd = VPDetection(length_thresh)
    estimated_once = False

    lp_path = str(save_dir)+'/LP/'
    vehicle_path = str(save_dir)+'/Vehicle/'
    if save_crop_lp: os.mkdir(lp_path)
    if save_crop: os.mkdir(vehicle_path)

    # Run tracking using the Strong sort algorithm
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        s = ''
        t1 = time_synchronized()

        if not estimated_once:
            vpd.find_vps(np.transpose(im, (1, 2, 0)))
            vps = vpd.vps_2D
            vp1 = vps[0]
            vp2 = vps[1]
            vp3 = vps[2]
            # vp1 = (1000, 800)
            # vp2 = (5000, 800)
            # vp3 = (320, 10000)
            estimated_once = True

        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_synchronized()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        
        # Non Max Supression is done by default in the algo
        pred = model(im,classes=classes)

        # Use this to provide custom values
        #pred=model.predict(im,classes=classes,iou=iou_thres,conf=conf_thres,agnostic_nms=agnostic_nms)

        t3 = time_synchronized()
        dt[1] += t3 - t2

        # This is for YOLO v7 object detection
        # Apply NMS
        #pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)

        dt[2] += time_synchronized() - t3
        
        data['frame_id']=frame_idx

        # Process detections
        for i, res_seg in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                #print("\n\npath:",p.name)
                #print("\n\n**************\n\n",p.stem)
                # video file
                if p.name.endswith(VID_FORMATS):#source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                    if save_crop_lp:
                        if not os.path.exists(lp_path + txt_file_name):os.mkdir(lp_path + txt_file_name) 
                    if save_crop:
                        if not os.path.exists(vehicle_path + txt_file_name):os.mkdir(vehicle_path + txt_file_name) 
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            #imc = im0.copy() if save_crop else im0  # for save_crop
            #imc_lp = im0.copy() if save_crop_lp else im0

            # doing deepcopy to get a new copy of image which any link the copied image
            imc_lp = copy.deepcopy(im0) #if save_crop_lp else im0
            imcopy = copy.deepcopy(im0) #this is to put the bounding boxes on the frame
            

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if res_seg.boxes is not None and len(res_seg.boxes):
                det = res_seg.boxes.data
                masks = res_seg.masks.data
                # print(det.shape)
                # print(masks.shape)
                # print(im0.shape)
                reshaped_mask = np.zeros((masks.shape[0], im0.shape[0], im0.shape[1]))
                for idx in range(len(masks)):
                    shape = masks[idx].cpu().numpy().shape
                    # print(im0.shape)
                    # print(shape)
                    fx = im0.shape[0] / shape[0] 
                    fy = im0.shape[1] / shape[1] 
                    # print(fx)
                    # print(fy)
                    resized = cv2.resize(masks[idx].cpu().numpy(), None, fx=fy, fy=fx, interpolation=cv2.INTER_CUBIC)  
                    reshaped_mask[idx] = resized

                #print('\n\nmasks:',len(masks))
                temp_det = det.clone()
                
                # Rescale boxes from img_size to im0 size
                temp_det[:, :4] = scale_coords(im.shape[2:], temp_det[:, :4], im0.shape).round()

                xywhs = xyxy2xywh(temp_det[:, 0:4])

                confs = temp_det[:, 4]
                clss = temp_det[:, 5]

                # pass detections to strongsort
                t4 = time_synchronized()
                output, mask = strongsort_list[i].update(xywhs.cpu(), torch.from_numpy(reshaped_mask), confs.cpu(), clss.cpu(), im0)
                outputs[i] = output
                frame_masks[i] = mask
                t5 = time_synchronized()
                dt[3] += t5 - t4

                # draw boxes for visualization
                
                if len(outputs[i]) > 0:
      
                    for j, (output, mask, conf) in enumerate(zip(outputs[i], frame_masks[i], confs)):
    
                        bboxes = output[0:4].astype(int)
                        id = int(output[4])
                        cls = int(output[5])
                        
                        #................................................................................#
                        # Speed Esitmation Logic
                        try:
                            fixp1, fixp2, fixp3 = get_contours(vp1, vp2, vp3, bboxes, mask)
                        except:
                            fixp1 = fixp2 = fixp3 = None
                            print("there is some error in the speed estimation")
                        # print(fixp1)
                        # print(fixp2)
                        # print(fixp3)
                        # assert False
                        if(save_txt):
                            bbox_left = int(output[0])
                            bbox_top = int(output[1])
                            bbox_w = int(output[2] - output[0])
                            bbox_h = int(output[3] - output[1])
                            data['vehc_bb'] = [bbox_left, bbox_top, bbox_w, bbox_h]
                        
                        data['vehc_id'] = id
                        data['vehc_cls'] = names[cls]    
                        data['vehc_conf'] = f"{conf:.2f}"

                        #print("\nstarting vehicle id:",id)
                        #print()
                        if fixp1 is not None:
                            if id in previous_pos:
                                # print(vp1)
                                # print(vp2)
                                # print(vp3)
                                vp1_left = line(vp1, np.array([0, im0.shape[0]]))
                                vp1_right = line(vp1, np.array([im0.shape[1], im0.shape[0]]))
                                if vp1[1] < im0.shape[0]:
                                    vp2_top = line(vp2, (vp1[0], vp1[1] + 100))
                                    vp2_bottom = line(vp2, np.array([im0.shape[1], im0.shape[0]]))
                                try:
                                    image_point = np.stack([intersection(vp1_left, vp2_top), intersection(vp1_left, vp2_bottom), intersection(vp1_right, vp2_top), intersection(vp1_right, vp2_bottom)]).astype(np.float32)
                                
                                    # This is for visualization
                                    # frame = np.array(im0)
                                    # i1 = intersection(vp1_left, vp2_top)
                                    # i2 = intersection(vp1_left, vp2_bottom)
                                    # i3 = intersection(vp1_right, vp2_top)
                                    # i4 = intersection(vp1_right, vp2_bottom)
                                    # frame = cv2.circle(frame, (int(i1[0]), int(i1[1])), 10, (255, 0, 0), thickness=10)
                                    # frame = cv2.circle(frame, (int(i2[0]), int(i2[1])), 10, (0, 255, 0), thickness=10)
                                    # frame = cv2.circle(frame, (int(i3[0]), int(i3[1])), 10, (0, 0, 255), thickness=10)
                                    # frame = cv2.circle(frame, (int(i4[0]), int(i4[1])), 10, (0, 255, 255), thickness=10)
                                    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                    # plt.show()
                                    # plt.savefig(f"intersect.png")
                                    # plt.close()
                                    # frame = pretty_line(frame, vp1, fixp1, (200, 200, 200), 1)
                                    # frame = pretty_line(frame, vp2, fixp2, (200, 200, 200), 1)
                                    # frame = pretty_line(frame, vp3, fixp3, (200, 200, 200), 1)
                                    
                                    
                                    to_point = np.array([[0, 0], [0, im0.shape[0]], [im0.shape[1], 0], [im0.shape[1], im0.shape[0]]]).astype(np.float32)
                                    M = cv2.getPerspectiveTransform(image_point, to_point)
                                    # frame = np.array(im0)
                                    # frame = cv2.warpPerspective(frame, M, [frame.shape[1] * 5, frame.shape[0]], flags=cv2.INTER_LINEAR)
                                    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                    # plt.show()
                                    # plt.savefig(f"warped.png")
                                    # plt.close()

                                    prev_bottom_pos = previous_pos[id]
                                    previous_pos[id] = fixp3
                                    p = M @ np.array([fixp3[0], fixp3[1], 1])
                                    p = p / p[2]
                                    p = p[:2]

                                    prev_p = M @ np.array([prev_bottom_pos[0], prev_bottom_pos[1], 1])
                                    prev_p = prev_p / prev_p[2]
                                    prev_p = prev_p[:2]

                                    frame_difference = frame_idx - previous_frames[id]
                                    previous_frames[id] = frame_idx
                                    p_prime = M @ np.array([fixp1[0], fixp1[1], 1])
                                    p_prime = p_prime / p_prime[2]
                                    p_prime = p_prime[:2]
                                    long_line = np.sqrt(np.sum(p - p_prime) ** 2)
                                    move_line = np.sqrt(np.sum(p - prev_p) ** 2)

                                    speed = move_line / long_line * 4.5 * fps / frame_difference

                                    mean_speed[id].append(speed)

                                    # y_min, x_min, y_max, x_max = boxes[i]
                                    data['vehc_speed'] = np.round(np.median(mean_speed[id]))
                                    #print(f"median speed: {data['vehc_speed']}")
                                except Exception as e:
                                    # Handle other exceptions
                                    print(f"Unexpected error: {e}")
                                    print("speed estimation has a problem")
                            else:
                                previous_pos[id] = fixp3
                                previous_frames[id] = frame_idx
                        #................................................................................#

                        # Padded resize
                        car_im0 = imcopy[bboxes[1]:bboxes[3],bboxes[0]:bboxes[2]]
                        #car_im0 = im0[bboxes[1]:bboxes[3],bboxes[0]:bboxes[2]]

                        car_im0_copy = imc_lp[bboxes[1]:bboxes[3],bboxes[0]:bboxes[2]]
                        
                        if 0 in car_im0_copy.shape:
                            continue
                        car_rs_img = letterbox(car_im0_copy, new_shape=imgsz, stride=stride)[0]
                        car_rs_img = car_rs_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                        car_rs_img = np.ascontiguousarray(car_rs_img)
                        

                        # Convert                       
                        car_rs_img = torch.from_numpy(car_rs_img).to(device)
                        car_rs_img = car_rs_img.half() if half else car_rs_img.float()  # uint8 to fp16/32
                        car_rs_img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if car_rs_img.ndimension() == 3:
                            car_rs_img = car_rs_img.unsqueeze(0)
                    
                        # Inference
                        lp_t1 = time_synchronized()
                        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                            lp_pred = lp_model(car_rs_img)[0]
                        lp_t2 = time_synchronized()

                        # Apply NMS
                        lp_pred = non_max_suppression(lp_pred, conf_thres, iou_thres, agnostic=agnostic_nms)
                        lp_t3 = time_synchronized()
                        
                        #................................................................................#
                        #Make and model 
                        image_mmr = Image.fromarray(car_im0_copy)
                        classname_matches = mmr_predict(image_mmr, mmr_model, mmr_img_transforms,device,mmr_df,mmr_k)
                        #print("car make and model:",classname_matches)
                        data['car_mmr']=result = "{}, {}, {}".format(*classname_matches)

                        classname_matches = mmr_predict(image_mmr, mmr_old_model, mmr_img_transforms,device,mmr_old_df,mmr_old_k)
                        #print("car make and model:",classname_matches)
                        data['car_sf_v_mmr']=result = "{}, {}, {}".format(*classname_matches)
                        #................................................................................#
                        
                        #................................................................................#
                        # Color Detection 

                        # 1st color detection method
                        color_histogram_feature_extraction.color_histogram_of_test_image(car_im0_copy)
                        prediction_1 = knn_classifier.main('color_recognition/training.data', 'test.data')
                        
                        # 2nd color detection method
                        user_image = ColorDetect(car_im0_copy)

                        # return dictionary of color count. Do anything with this
                        # Sample output: How much percentage of the image consits of which color
                        try:
                            prediction_2 = user_image.get_color_count()
                            #print(f"the color of the vehicle can be either of these: {prediction_1} or {prediction_2}")
                            data['vehc_color']=[prediction_1,prediction_2]
                        except Exception as e:
                            print("there is some error in PREDICTION_2 of color model: ", e)
                            prediction_2 = 'NA'
                            #print(f"the color of the vehicle can be either of these: {prediction_1} or {prediction_2}")
                            data['vehc_color']=[prediction_1,prediction_2]
                        #................................................................................#

                        #................................................................................#
                        # Process lp_detections
                        for lp_i, lp_det in enumerate(lp_pred):  # detections per image
                            #lp_det is a list of lists [[bb,conf,class]] so 6 elements in the list                         
                            
                            if len(lp_det):
                                # Rescale boxes from img_size to im0 size
                                lp_gn = torch.tensor(car_im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                                lp_det[:, 0:4] = scale_coords(car_rs_img.shape[2:], lp_det[:, :4], car_im0.shape).round() #xyxy

                                cls_lp = lp_det[0][5]
                                conf_lp = f"{lp_det[0][4]:.2f}"
                                
                                xyxy_lp = lp_det[0][0:4].detach().cpu().numpy()
                                lp_xmin,lp_ymin,lp_xmax,lp_ymax = xyxy_lp.astype(int)

                                label_lp = f"LP {conf_lp}"
                                
                                data['LP_conf'] = conf_lp

                                #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                # LP crop extracted from copy of the vehicle crop
                                lp_im0 = car_im0_copy[lp_ymin:lp_ymax,lp_xmin:lp_xmax]                    
                                
                                # Applying SR before passing to the OCR
                                try:
                                    sr_lp, _ = upsampler.enhance(lp_im0, outscale=opt.outscale)
                                except Exception as e:
                                    sr_lp = lp_im0
                                    print('Error', e)
                                    print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
                                
                                #easyocr_result = easyocr_reader.recognize(lp_im0)#,paragraph="False")   
                                easyocr_result = easyocr_reader.readtext(sr_lp)     #readtext works better than recognize 
                                #print("\neasy ocr result:",easyocr_result)                                    
                                ocr_label = " "    
                                ocr_conf = " "  

                                # Concatinating the OCR results of a license plate
                                for res in easyocr_result:
                                    if res[2] > 0.1: #only considering the output for which the confidence is more than 10%
                                        ocr_label += res[1] + ";"
                                        ocr_conf += f'{res[2]:.2f};'
                                  
                                if ocr_label == ' ':
                                    data['LP_txt']= 'NA'
                                    data['LP_txt_conf']= 'NA'
                                else:
                                    data['LP_txt']= ocr_label
                                    data['LP_txt_conf']= ocr_conf
                                
                                 
                                label_lp = label_lp + ocr_label
                                #print("label per detection :",label_lp)
                                # Ploting bounding box of the LP
                                plot_one_box(lp_det[0][0:4], car_im0, label=label_lp, color=colors[int(cls_lp)], line_thickness=1)
                            
                                # Write results
                                if save_txt:  # Write to file
                                    #xyxy_lp = lp_det[:, 0:4]
                                    xyxy2xywh(torch.tensor(xyxy_lp).view(1, 4))[0]
                                    xywhs_lp = xyxy2xywh(torch.tensor(xyxy_lp).view(1, 4))[0].detach().cpu().numpy().astype(int)
                                    
                                    data['LP_bb'] = xywhs_lp.tolist()
                                    

                                    with open(txt_path + '.txt', 'a') as f:
                                        json.dump(data, f, ensure_ascii=False)
                                        f.write('\n')

                                if save_crop_lp:                            
                                    cv2.imwrite(lp_path+txt_file_name+'/'+ str(frame_idx) +"_"+ str(id)+ocr_label+'.jpg',sr_lp)

                                #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                            else:
                                data['LP_bb']='NA'
                                data['LP_conf']='NA'
                                data['LP_txt']='NA'
                                data['LP_txt_conf']='NA'
                                with open(txt_path + '.txt', 'a') as f:
                                        json.dump(data, f, ensure_ascii=False)
                                        f.write('\n')
                        #................................................................................#
 
                        if save_vid or save_crop or show_vid:  # Add bbox to the video frame

                            label = None if hide_labels else (f'{id} {names[cls]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[cls]} {conf:.2f}'))
                            #plot_one_box(bboxes, im0, label=label, color=colors[cls], line_thickness=1.5)
                            
                            plot_one_box(bboxes, imcopy, label=label, color=colors[cls], line_thickness=1)


                            if save_crop:
                                #if not os.path.exists(vehicle_path + txt_file_name):os.mkdir(vehicle_path + txt_file_name) 
                                cv2.imwrite(vehicle_path+txt_file_name+'/' + str(frame_idx)+"_"+label+'.jpg',car_im0_copy)

                                

                    # assert False                   
                      
                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                print('No detections')
                data = {'frame_id': frame_idx, 'vehc_id':-1, 'vehc_cls':'NA', 'vehc_bb':'NA', 'vehc_conf':0,'vehc_color':'NA', 'car_mmr':'NA','car_sf_v_mmr':'NA', 'LP_bb':'NA', 'LP_conf':0, 'LP_txt':'NA', 'LP_txt_conf':0, 'vehc_speed':0}
                

            # Stream results on the output video
            if show_vid:
                #cv2.imshow(str(p), im0)
                cv2.imshow(str(p), imcopy)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                #vid_writer[i].write(im0)
                vid_writer[i].write(imcopy)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, imgsz, imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')

    parser.add_argument('--save-crop-lp', action='store_true', help='save cropped LP prediction boxes')

    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    #SR -------------------------------------------------------------------------------------------
    parser.add_argument('--denoise_strength', type=float, default=0.5,help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. Only used for the realesr-general-x4v3 model'))
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision). Required to be true for CPUs!')
    parser.add_argument('--alpha_upsampler', type=str, default='realesrgan',help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus',help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x2plus | realesr-general-x4v3'))
    #SR -------------------------------------------------------------------------------------------
    parser.add_argument("--fps", type=int, default=24)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
