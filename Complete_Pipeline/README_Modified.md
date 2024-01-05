#Authors :Jahnavi and Samridhi

Model Used: YOLOv8 seg + Resnet for Make and model + Fine Tuned EasyOCR + RealESGAN Super Resolution (our best performing model) + Speed estimation + color detection

Note, the accuracy for LP detection reported in the paper is based on the testing dataset of the UFPR-dataset which we aquired on request from https://github.com/raysonlaroca/ufpr-alpr-dataset. In this repository we share a video file as an example to observe the working of our proposed mdoel as the dataset itself was about 10GB. We did however added a subset of the UFPR testing dataset here in the test folder. 

Our LP code was built on top of the repository https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet.git, the pretrained lp detection file found in the weights folder is taken from https://github.com/ANPR-ORG/ANPR-using-YOLOV7-EasyOCR and 

the pretrained make and model weights found in the git repo https://github.com/Pells31/Vehicle-Make-and-Model-Recognition.

The color detection is a combination of two codes: https://github.com/MarvinKweyu/ColorDetect and https://github.com/ahmetozlu/color_recognition

Speed estimation code was build on top of the repo ....... (Please get this information from Tong)



We provide two models here for testing and reproducing our results: 
1. Test with video and tracking
    One Video file example has been added 
2. Test with image dataset (*These are the results reported in our paper for LP detection)
    Testing Datasets from the UFPR (see the test folder)


HOW TO RUN OUR CODE: 

1. To create a new environment: 
conda create --name alprenv
conda activate alprenv

2. pip3 install -r requirements.txt

3. In the Yolov7_StrongSORT_OSNet directory, perform this : 

mkdir ~/.EasyOCR
cd ~/.EasyOCR
mkdir model
mkdir user_network
cp Yolov7_StrongSORT_OSNet/custom_example.py ~/.EasyOCR/user_network
cp Yolov7_StrongSORT_OSNet/custom_example.yaml ~/.EasyOCR/user_network
cp Yolov7_StrongSORT_OSNet/custom_example.pth ~/.EasyOCR/model

4. Run this command for the output: 

For Video Input Data

#first go to rtx node in frontera with the following command and then run your python jobs
##idev -N 1 -n 1 -p rtx-dev -t 02:00:00


With GPU available:
# Please run this, this command is for new YOLO Model
python3 main.py --conf-thres 0.25 --source test/20230222_115854.mp4 --device 0 --save-crop-lp --save-crop --save-vid --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov8n-seg.pt --classes 1 2 3 5 7


With CPU: 
# This command is for new YOLO Model
python3 main.py --conf-thres 0.25 --source test/20230222_115854.mp4 --device 'cpu' --save-crop-lp --save-crop --save-vid --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov8n-seg.pt --classes 1 2 3 5 7

The Output produced has the following format: (As an example see exp4 in runs/track)
1. 20230222_120908.mp4	[This output video has tracking, bounding box, class, and predicted OCR output information as the car is moving]
2. Vehicle [Cropped Vehicle images]
3. LP [Cropped license plates with super resolution]
4. tracks [Predicted output results of every vehicle for every frame]



# I do not think the following things are re
For Image Input Data

With GPU: 
python3 yolov7/test_mmr_lp_SR.py --source test/track0096 --weights weights/yolov7.pt --conf 0.25 --save-lp --save-txt-lp --nosave --save-conf --device 0 --classes 1 2 3 5 7 --save-txt

With CPU: 
python3 yolov7/test_mmr_lp_SR.py --source test/track0096 --weights weights/yolov7.pt --conf 0.25 --save-lp --save-txt-lp --nosave --save-conf --device 'cpu' --classes 1 2 3 5 7 --save-txt

Try it out with different track folders as you may like available in the test folder!


The Output produced has the following format: (As an example see exp in runs/test)
1. LP [Cropped LP with super resolution]
2. correct_lp_text.txt	[Correct OCR predictions] Correct for us means when the true text is present somewhere in the predicted text.
3. test.txt [Accuracy results and the IOU that was less than 0.7]
4. correct_lp_bb.txt 
5. labels.txt [LP output for each detected LP with confidence, predicted value, and true value]
6. mmr.txt [make and model predictions], please note that we consider the predicted output to be correct when the true make is in the predicted make and model output
7. correct_mmr.txt

