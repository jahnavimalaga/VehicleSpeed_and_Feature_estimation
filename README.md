# Models Used
YOLOv8n-seg + Resnet for Make and model + Fine Tuned EasyOCR + RealESGAN Super Resolution (our best-performing model) + Histogram & KNN and ColorDetect for color detection + Speed Estimation logic proposed by Dubska et al. (2014)

**Note**: The accuracy for LP detection reported in the paper is based on the testing dataset of the UFPR-dataset, acquired from [UFPR ALPR Dataset](https://github.com/raysonlaroca/ufpr-alpr-dataset). In this repository, we share a video file as an example to observe the working of our proposed model as the dataset itself was about 10GB.

Our code was built on top of the repository [Yolov7_StrongSORT_OSNet](https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet). The pre-trained LP detection file found in the weights folder is taken from [ANPR-ORG/ANPR-using-YOLOV7-EasyOCR](https://github.com/ANPR-ORG/ANPR-using-YOLOV7-EasyOCR), and the pre-trained make and model weights are from [Vehicle-Make-and-Model-Recognition](https://github.com/Pells31/Vehicle-Make-and-Model-Recognition), fine-tuned make and model weights are from [https://github.com/lamhagoel/Vehicle-Make-and-Model-Recognition.git], object detection weights are from [https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt] and speed estimation is built based on Dubská 2014.

## Provided Models for Testing and Reproducing Results

Note: do <<pip freeze > requirements.txt>> to freeze the env

1. **Test with video and tracking**
   - One video file example has been added

2. **Test with image dataset (*Results reported in our paper for LP detection)**
   - Testing Datasets from the UFPR
   - **Note**: Haven't tested with images recently; may not work as expected.

## How to Run Our Code
Note: do <<pip freeze > requirements.txt>> to freeze the env

1. **To create a new environment:**
   ```bash
   conda create --name test_vvd python=3.8
   conda activate test_vvd
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt

Run this command if torch is not already installed:
   ```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
3. In the pwd directory, perform this:
```bash
mkdir ~/.EasyOCR
cd ~/.EasyOCR
mkdir model
mkdir user_network
cp Complete_Pipelline/custom_example.py ~/.EasyOCR/user_network
cp Complete_Pipelline/custom_example.yaml ~/.EasyOCR/user_network
cp Complete_Pipelline/custom_example.pth ~/.EasyOCR/model
```
4. Download the .pt files as instructed in folders in the files Complete_Pipeline/weights/file_to_download.txt and Complete_Pipeline/yolov7/models/files_to_download.txt

### 5. You can use any of the following 3 ways to run analysis on Video Input Data
A - To automatically run the job i.e to download the file from GDrive, run the algo, and upload the processed results to box and gdrive.

1. First set up the rclone and link the online storage location accordingly
2. Change the environment name accordingly in the run.sh if you are using a different env_name and run the below command
```bash
run.sh
```

B - Go to rtx node in frontera with the following command and then run your python jobs
```bash 
idev -N 1 -n 1 -p rtx-dev -t 02:00:00
```
With GPU available:
``` bash
python3 main.py --conf-thres 0.25 --source test/20230222_115854.mp4 --device 0 --save-crop-lp --save-crop --save-vid --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov8n-seg.pt --classes 1 2 3 5 7
python post_processing.py --path <<path to the file or track directory>>
```
With CPU:
```bash
# Have not tested this recently
python3 main.py --conf-thres 0.25 --source test/20230222_115854.mp4 --device 'cpu' --save-crop-lp --save-crop --save-vid --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov8n-seg.pt --classes 1 2 3 5 7
python post_processing.py --path <<path to the file or track directory>>
```
C - It can also be run with script.slurm file by running the below commands
Change the path in the script.slurm file accordingly
```bash
# 1. conda activate <<env name>>
# 2. sbatch script.slurm
```

## Output 

Output has the following format: (As an example see in runs/track)

xxx.mp4 - This output video has tracking, bounding box, class, color, make & model, speed and predicted OCR output information as the car is moving.
Vehicle - Cropped Vehicle images (if mentioned)
LP - Cropped license plates with super resolution (if mentioned)
tracks - Predicted output results of every vehicle for every frame

## References:
Dubská, M., Herout, A., & Sochor, J. (2014). Automatic Camera Calibration for Traffic21 Understanding, In BMVC (Vol. 4, No. 6, p. 8). PDF




