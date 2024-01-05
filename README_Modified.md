#Authors :Jahnavi

Model Used: YOLOv8n-seg + Resnet for Make and model + Fine Tuned EasyOCR + RealESGAN Super Resolution (our best performing model) + Histogram & KNN and ColorDetect for color detection + Speed Estimation logic proposed by Dubska et al. (2014)

Note, the accuracy for LP detection reported in the paper is based on the testing dataset of the UFPR-dataset which we aquired on request from https://github.com/raysonlaroca/ufpr-alpr-dataset. In this repository we share a video file as an example to observe the working of our proposed model as the dataset itself was about 10GB. We did however added a subset of the UFPR testing dataset here in the test folder. 

Our code was built on top of the repository https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet.git, the pretrained lp detection file found in the weights folder is taken from https://github.com/ANPR-ORG/ANPR-using-YOLOV7-EasyOCR and the pretrained make and model weights found in the git repo https://github.com/Pells31/Vehicle-Make-and-Model-Recognition.

We provide two models here for testing and reproducing our results: 
1. Test with video and tracking
    One Video file example has been added 

# Have not tested this recently, no this may not work as expected
2. Test with image dataset (*These are the results reported in our paper for LP detection)
    Testing Datasets from the UFPR (see the test folder)


HOW TO RUN OUR CODE: 
Note: do <<pip freeze > requirements.txt>> to freeze the env

1. To create a new environment: 
    conda create --name test_vvd python=3.8
    conda activate test_vvd

2. pip3 install -r requirements.txt
   Run this command if troch is not already installed -> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3. In the Complete_Pipelline directory, perform this : 

mkdir ~/.EasyOCR
cd ~/.EasyOCR
mkdir model
mkdir user_network
cp Yolov7_StrongSORT_OSNet/custom_example.py ~/.EasyOCR/user_network
cp Yolov7_StrongSORT_OSNet/custom_example.yaml ~/.EasyOCR/user_network
cp Yolov7_StrongSORT_OSNet/custom_example.pth ~/.EasyOCR/model

4. Run this command for the output: 

For Video Input Data



# You can use any of the following 3 ways to run analysis

# A - To automatically run the job i.e download the file from gdrive, run the algo and upload the processed results
# Before that first set up the rclone and link the online storage location accordingly
# Then change the environment name accordingly in the run.sh
1. run.sh

# B - First go to rtx node in frontera with the following command and then run your python jobs
idev -N 1 -n 1 -p rtx-dev -t 02:00:00
# With GPU available:
# Please run this, this command is for new YOLO Model
python3 main.py --conf-thres 0.25 --source test/20230222_115854.mp4 --device 0 --save-crop-lp --save-crop --save-vid --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov8n-seg.pt --classes 1 2 3 5 7
python post_processing.py --path <<path to the file or track directory>>

# With CPU: 
# This command is for new YOLO Model, have not tested this
python3 main.py --conf-thres 0.25 --source test/20230222_115854.mp4 --device 'cpu' --save-crop-lp --save-crop --save-vid --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov8n-seg.pt --classes 1 2 3 5 7
python post_processing.py --path <<path to the file or track directory>>


# C - It can also be run with script.slurm file by running the below commands
change the path in the script.slurm file accordingly
1. conda activate <<env name>>
2. sbatch script.slurm 

The Output produced has the following format: (As an example see exp4 in runs/track)
1. 20230222_120908.mp4	[This output video has tracking, bounding box, class, and predicted OCR output information as the car is moving]
2. Vehicle [Cropped Vehicle images] if it is mentioned
3. LP [Cropped license plates with super resolution] if it is mentioned
4. tracks [Predicted output results of every vehicle for every frame]



Dubská, M., Herout, A., & Sochor, J. (2014). Automatic Camera Calibration for Traffic21 Understanding, In BMVC (Vol. 4, No. 6, p. 8). Available at:22 https://www.fit.vutbr.cz/~herout/papers/2014-BMVC-VehicleBoxes.pdf