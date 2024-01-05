#!/usr/bin/env python3

"""
    Modified from https://github.com/mikel-brostrom/yolov7/blob/main/test.py
    Updated by Jahnavi Malagavalli (jm222479), April 2023

    Vehicle & License Plate detection test script using YOLO, Real-ESRGAN, EasyOCR

    Usage:   python3 yolov7/test_mmr_lp_SR.py --source test/track0106 --weights weights/yolov7.pt --conf 0.25 --save-lp --save-txt-lp --nosave --save-conf --device 0 --classes 1 2 3 5 7 --save-txt

    #observation - 
    some charecters are detected as small letters instead of capitals and they are not counted as correct label
"""

import argparse
import time
from pathlib import Path

import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path,box_iou
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import easyocr

import os
import sys

import re


#----------------------------------------for SR
import glob
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
#-----------------------------------------

# make and model
import pandas as pd
from predict_mmr import mmr_predict,initialize_model,Image,transforms
#

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov7 strongsort root directory


if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
#if str(ROOT / 'yolov7') not in sys.path:
#    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH


# Code for downloading the super resolution model
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

    # use dni to control the denoise strength
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

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA


    # Load model
    model = attempt_load(Path(weights[0]), map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    #lp detection weights
    lp_weights, trace =  "weights/lp_best.pt", False
    lp_model = attempt_load(Path(lp_weights), map_location=device)  # load FP32 model
    stride = int(lp_model.stride.max())  # model stride
    if half:  lp_model.half()

        # Run inference
    if device.type != 'cpu':
        lp_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(lp_model.parameters())))  # run once

    save_lp = opt.save_lp#True
    save_vehicle = opt.save_vehicle#True

    #This is the fine tuned OCR model stored in ~/.EasyOCR
    easyocr_reader = easyocr.Reader(['en'],recog_network='custom_example')
    
    lp_path = str(save_dir)+'/LP/'
    vehicle_path = str(save_dir)+'/Vehicle/'

    if save_lp: os.mkdir(lp_path)
    if save_vehicle: os.mkdir(vehicle_path)

    #-----------------
    correct_lp_bb = 0
    correct_lp_txt = 0
    correct_make_txt = 0
    #-----------------

    #SR---------------
    upsampler = SR(opt)
    #SR---------------

    # For make and model--------------------------------------
    model_name="resnet50_40epochs_mmr.pt"
    mmr_k=3
    #classname_matches = predict(path, model_name, k)
    #print(classname_matches)
    #img = Image.open(path)
    #if img.mode != "RGB":  # Convert png to jpg
    #    img = img.convert("RGB")
    mmr_img_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    mmr_df = pd.read_pickle("yolov7/data/preprocessed_data_mmr.pkl")
    mmr_num_classes = mmr_df["Classname"].nunique()
    print("model name:",model_name[:8])
    mmr_model, _ = initialize_model(model_name[:8], mmr_num_classes, feature_extract=True)
    path = os.path.dirname(__file__)
    mmr_model.load_state_dict(torch.load(path + "/models/" + str(model_name), map_location=device))
    mmr_model.to(device)
    mmr_model.eval()
    #--------------------------------------

    t0 = time.time()
    for frame_idx,(path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Extracting true labels of LP text, and bounding box from the testing folders
        label_path = path.replace('png','txt')
        lp_labels = open(label_path,"r")
        lp_data = lp_labels.readlines()

        for lp_line in lp_data:
            if "plate" in lp_line:
                if 'position' in lp_line:
                    lst_bb = lp_line.split(' ')
                    lst_bb = lst_bb[1:]
                    lst_bb[-1] = lst_bb[-1].replace('\n','')

                    true_lp_xyxy = torch.tensor([[int(lst_bb[0]),int(lst_bb[1]),int(lst_bb[0])+int(lst_bb[2]),int(lst_bb[1])+int(lst_bb[3])]]).to(device)
                    break
                else:
                    true_lp_txt = lp_line.split(':')[1]
                    true_lp_txt = true_lp_txt.replace('\n','')  
                    true_lp_txt = re.sub('[^A-Za-z0-9]+', '', true_lp_txt)
            if 'make' in lp_line:
                a = lp_line.split(':')
                true_make_txt = lp_line.split(':')[1]
                #print("\n\ntrue make:",a[1])
                true_make_txt = true_make_txt.replace(' ','')
                true_make_txt = true_make_txt.replace('\n','')
                #true_make_txt = re.sub('[^A-Za-z0-9]+', '', true_lp_txt)       
                #print("\n\ntrue make:",true_make_txt)

        lp_labels.close()


        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        #-------------------------------------------
        #LP detection
        save_crop_lp = True
        imc_lp = im0s.copy() if save_crop_lp else im0
        #-------------------------------------------
        txt_path = str(save_dir / 'labels')
        # Process detections, pred is detection of multiple objects
        # this is nothing just [[det]]
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                #detection per images is this
                for vehc_id,vehc_bb in enumerate(det):
                    #................................................................................#
                    #print(f"detection:{det} vehid:{vehc_id}")
                    car_im0 = im0[int(vehc_bb[1]):int(vehc_bb[3]),int(vehc_bb[0]):int(vehc_bb[2])]
                    car_im0_copy = imc_lp[int(vehc_bb[1]):int(vehc_bb[3]),int(vehc_bb[0]):int(vehc_bb[2])]
                    
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
                    #lp_t1 = time_synchronized()
                    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                        lp_pred = lp_model(car_rs_img)[0]
                    #lp_t2 = time_synchronized()

                    # Apply NMS
                    lp_pred = non_max_suppression(lp_pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)

                    #Make and model------------------------------------------------------------------
                    image_mmr = Image.fromarray(car_im0_copy)
                    classname_matches = mmr_predict(image_mmr, mmr_model, mmr_img_transforms,device,mmr_df,mmr_k)
                    print("car make and model:",classname_matches)
                    if save_txt:
                        with open(str(save_dir) + '/mmr.txt', 'a') as f:
                            #f.write(" "+str(p.stem)
                            f.write(f'{frame_idx+1,p.stem,vehc_id,classname_matches} true make {true_make_txt}\n')
                    classname_matches_str = ' '.join(classname_matches)
                    if true_make_txt in classname_matches_str:
                        correct_make_txt+=1
                        with open(str(save_dir) + '/correct_mmr.txt', 'a') as f:
                            #f.write(" "+str(p.stem)
                            f.write(f'{frame_idx+1,p.stem,vehc_id,classname_matches_str} true make {true_make_txt}\n')
                    #------------------------------------------------------------------
                    # Process lp_detections                    
                    for lp_i, lp_det in enumerate(lp_pred):  # detections per cropped vehicle image                                                    
                        if len(lp_det):
                            # Rescale boxes from img_size to im0 size
                            lp_gn = torch.tensor(car_im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            lp_det[:, 0:4] = scale_coords(car_rs_img.shape[2:], lp_det[:, :4], car_im0.shape).round() #xyxy
        
                            cls_lp = lp_det[0][5]
                            conf_lp = lp_det[0][4]
                            
                            label_lp = f"LP {conf_lp:.2f}"
                            # Print results
                            lp_s = ''
                            for lp_c in lp_det[:, -1].unique():
                                lp_n = (lp_det[:, -1] == lp_c).sum()  # detections per class
                                lp_s += f"{n} {names[int(lp_c)]}{'lp_s' * (lp_n > 1)}, "  # add to string
    
                            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                            
                            #Extracting Lp img
                            lp_im0 = car_im0_copy[int(lp_det[0][1]):int(lp_det[0][3]),int(lp_det[0][0]):int(lp_det[0][2])] 

                            #SR-----------------------------------------------------
                            try:
                                sr_lp, _ = upsampler.enhance(lp_im0, outscale=opt.outscale)
                            except RuntimeError as error:
                                sr_lp = lp_im0
                                print('Error', error)
                                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')


                            # sr_lp is the lp image after applying super resolution   
                            easyocr_result = easyocr_reader.readtext(sr_lp)     #readtext works better thatn recognize                                     
                            ocr_label = " "  

                            ocr_label_temp = ''
                            #doing for each license plate
                            for res in easyocr_result:
                                if res[2] > 0.1:
                                    ocr_label += res[1] + " "+f'{res[2]:.2f} '
                                    ocr_label_temp += res[1]
                                    '''
                                    #-----------------------------------------------------
                                    #this regular expression only return alphabets and numbers
                                    if true_lp_txt in re.sub('[^A-Za-z0-9]+', '', res[1]):
                                        correct_lp_txt+=1
                                    #-----------------------------------------------------
                                    '''
                                    
                            #-----------------------------------------------------
                            #this regular expression only return alphabets and numbers
                            if true_lp_txt in re.sub('[^A-Za-z0-9]+', '', ocr_label_temp):
                                correct_lp_txt+=1
                                with open(str(save_dir) + '/correct_lp_text.txt', 'a') as f_test1:
                                    f_test1.write(f"{frame_idx+1} {vehc_id} predicted_txt: {ocr_label_temp} true_txt :{true_lp_txt} + \n")
                            #-----------------------------------------------------
                         
                            label_lp = label_lp + ocr_label
                            print("label per detection :",label_lp)
                            plot_one_box(lp_det[0][0:4], car_im0, label=label_lp, color=colors[int(cls_lp)], line_thickness=2)
                            # Write results

                            #----------------------------------------------
                            lp_org_bb =  torch.tensor([[lp_det[0][0]+vehc_bb[0],lp_det[0][1]+vehc_bb[1],lp_det[0][2]+vehc_bb[0],lp_det[0][3]+vehc_bb[1]]]).to(device)
                            ious= box_iou(lp_org_bb[:,:4], true_lp_xyxy)  # best ious

                            #Choosing the detected LP's with more than 0.7 intersection union value with the true LP Bounding box
                            if ious[0][0]>0.7:
                                correct_lp_bb+=1
                                with open(str(save_dir) + '/correct_lp_bb.txt', 'a') as f_test1:
                                    f_test1.write(f"{frame_idx+1} {vehc_id} correct_lp :{correct_lp_bb} + \n")
                            else:
                                with open(str(save_dir) + '/test.txt', 'a') as f_test:
                                    f_test.write(f"{frame_idx+1} {vehc_id} iou :{ious[0][0]} + \n")
                            #----------------------------------------------

                            for *xyxy_lp, conf_lp, cls_lp in reversed(lp_det):
                                if opt.save_txt_lp:  # Write to file
                                    xywhs_lp = xyxy2xywh(torch.tensor(xyxy_lp).view(1, 4))[0]
                                    
                                    #Writing LP bounding box and frame/image id
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * 5) % (frame_idx + 1,xywhs_lp[0],xywhs_lp[1],xywhs_lp[2],xywhs_lp[3]))
                                        f.write(" "+str(p.stem) +" " + label_lp +'true label - ' + true_lp_txt  + '\n')
                                if save_lp: #saving cropped after SR LP img
                                    cv2.imwrite(lp_path+str(frame_idx+1)+'_'+str(vehc_id)+"_"+ocr_label+'_sr.jpg',sr_lp)
                                    
                            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                            #................................................................................# 


                    # Write results
                    xyxy = vehc_bb[:4]
                    conf = vehc_bb[4] 
                    cls = vehc_bb[5] #class of the vehicle object
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(xyxy.clone().detach().view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (frame_idx + 1,vehc_id,cls, *xywh, conf) if opt.save_conf else (frame_idx + 1,vehc_id,cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line)
                            f.write(" "+ f'{names[int(cls)]}' + '\n')

                    if save_img or view_img:  # Add bbox to image
                        
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    if save_vehicle:                  
                        cv2.imwrite(vehicle_path + str(frame_idx+1)+'_'+str(vehc_id)+"_"+label+'.jpg',car_im0_copy)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (w, h))
                    vid_writer.write(im0)

    #if save_txt or save_img:
    #    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''

    dataset_length = len(dataset)
    print(f"correct lp bb count:{correct_lp_bb}/{dataset_length}")
    print(f"correct lp txt count:{correct_lp_txt}/{dataset_length}")
    print(f"correct make txt count:{correct_make_txt}/{dataset_length}")
    with open(str(save_dir) + '/test.txt', 'a') as f_test:
        f_test.write(f"correct lp bb count:{correct_lp_bb}/{dataset_length}")
        f_test.write(f"\ncorrect lp txt count:{correct_lp_txt}/{dataset_length}")
        f_test.write(f"\ncorrect make txt count:{correct_make_txt}/{dataset_length}")

    print(f'Done. ({time.time() - t0:.3f}s)')

   
def argsparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/test', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--save-lp', action='store_true', help='save lp of each vehicle')
    parser.add_argument('--save-vehicle', action='store_true', help='save cropped vehicle images')
    parser.add_argument('--save-txt-lp', action='store_true', help='save results to *.txt')

    #SR -------------------------------------------------------------------------------------------
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus',
                        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x2plus | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-dn', '--denoise_strength', type=float, default=0.5,
                        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
                              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--fp32', action='store_true',
                        help='Use fp32 precision during inference. Default: fp16 (half precision). Required to be true for CPUs!')
    parser.add_argument('--alpha_upsampler', type=str, default='realesrgan',
                        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    #SR -------------------------------------------------------------------------------------------

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()

    opt = argsparse()#parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']: #Using Yolov7 for obj detection
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
