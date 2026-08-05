
## Setting Up

Download training data (object detection only)
```shell
aidata download dataset \
 --config https://docs.mbari.org/internal/ai/projects/config/config_uav.yml \
 --base-path $PWD/TrainNegatives07152026 \
 --voc \
 --labels "Batray","Bird","Boat","Cement_Ship","Egregia","Fish","Jelly","Kayak","Kelp","Mola","Mooring_Buoy","Otter","Person","Pinniped","Secci_Disc","Shark","Surfboard","Velella_velella","Velella_velella_raft","Whale"  \
 --verified \
 --token $TATOR_TOKEN \
 --disable-ssl-verify  \
 --exclude-versions testset --single-class object
 ```

 Download testing data (object detection only)
 ```shell
 aidata download dataset \
 --config https://docs.mbari.org/internal/ai/projects/config/config_uav.yml \
 --base-path $PWD/Test07152026 \
 --voc \
 --labels "Batray","Bird","Boat","Cement_Ship","Egregia","Fish","Jelly","Kayak","Kelp","Mola","Mooring_Buoy","Otter","Person","Pinniped","Secci_Disc","Shark","Surfboard","Velella_velella","Velella_velella_raft","Whale"  \
 --verified \
 --token $TATOR_TOKEN \
 --disable-ssl-verify  \
 --version testset 
 ```

Transform the training data. For YOLO models (all sizes), use 640x640 images. 
```shell
aidata transform voc --base-path $PWD/TrainNegatives07152026 --resize 640 --crop-size 640  --crop-overlap 0.5 --min-visibility 0.0 --min-dim 20 --negative-percent 0.2
```

For RF-DETR Medium use 576x576 crops. For other RF-DETR sizes, see [github](https://github.com/roboflow/rf-detr):
```shell
aidata transform voc --base-path $PWD/TrainNegatives07152026 --resize 576 --crop-size 576  --crop-overlap 0.5 --min-visibility 0.0 --min-dim 20
```

Include negative crops with --negative-percent, recommended:
```shell
aidata transform voc --base-path $PWD/TrainNegatives07152026 --resize 576 --crop-size 576  --crop-overlap 0.5 --min-visibility 0.0 --min-dim 20 --negative-percent 0.05
```

VOC to YOLO format transform:
```shell
aidata transform voc-to-yolo  --base-path $PWD/TrainNegatives07152026/transformed
```

Split the data:
```shell
aidata transform split -i $PWD/TrainNegatives07152026/transformed -o $PWD/TrainNegatives07152026split --split 0.9,0.1,0.0
```

Compress the testing data for easier upload into Drive:
```shell
tar -czf images.tar.gz -C $PWD/Test07152026/testset images
tar -czf labels.tar.gz -C $PWD/Test07152026/testset labels
```


## Training an RF-DETR model in Google Colab Environment

Create data.yaml file
```shell
train: /content/datasets/train/images
val: /content/datasets/valid/images
test: /content/datasets/test/images

nc: 1
names: ['object']
```

Upload data.yaml file, and image/label tar files to Google Drive. Google Drive for Desktop makes this easy

In a Colab notebook:
Install roboflow and dependencies
```shell
%pip install "rfdetr[train,loggers]==1.7.1" -q
%pip install supervision roboflow -q
```
 
Set up HOME
```shell
import os
HOME = os.getcwd()
print(HOME)
```
Mount to drive 
```shell
# Allow access to personal google drive and add new folders

# Connect Google Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=True) # This will prompt for authorization.

# This will create the uavs files if they don't exist.
folders =  ["uavs/"]
for folder in folders:
  path = "/content/drive/MyDrive/" + folder
  if not os.path.exists(path): # Create the folder if it does not exist
    os.mkdir(path)
```

Move in the compressed files and unpack them.
```shell
!mkdir {HOME}/datasets
%cd {HOME}/datasets
from google.colab import userdata
uavs_folder = "/content/drive/MyDrive/uavs/"

!mkdir -p /content/datasets/savedir/
!cp -r "/content/drive/MyDrive/uavs/images.tar.gz" "/content/datasets/savedir/"

!cp -r "/content/drive/MyDrive/uavs/labels.tar.gz" "/content/datasets/savedir/"

!tar xf /content/datasets/savedir/images.tar.gz --directory /content/datasets/savedir/

!tar xf /content/datasets/savedir/labels.tar.gz --directory /content/datasets/savedir/
```

Move the data to the directory that RF-DETR expects
```shell
## create directories
!mkdir /content/datasets/train/
!mkdir /content/datasets/train/images/
!mkdir /content/datasets/train/labels/
!mkdir /content/datasets/test/
!mkdir /content/datasets/test/images/
!mkdir /content/datasets/test/labels/
!mkdir /content/datasets/valid/
!mkdir /content/datasets/valid/images/
!mkdir /content/datasets/valid/labels/

#get the data.yaml file
!cp "/content/drive/MyDrive/uavs/data.yaml" "/content/datasets/data.yaml"
!ls /content/datasets/

#move the data to the expected directories
!cp -r "/content/datasets/savedir/images/train/." "/content/datasets/train/images/"
!cp -r "/content/datasets/savedir/labels/train/." "/content/datasets/train/labels/"

!cp -r "/content/datasets/savedir/images/test/." "/content/datasets/test/images/"
!cp -r "/content/datasets/savedir/labels/test/." "/content/datasets/test/labels/"

!cp -r "/content/datasets/savedir/images/val/." "/content/datasets/valid/images/"
!cp -r "/content/datasets/savedir/labels/val/." "/content/datasets/valid/labels/"

!ls /content/datasets/
```

Train a new RF-DETR Nano model:
```shell
from rfdetr import RFDETRNano

model = RFDETRNano()

model.train(
    dataset_dir="/content/datasets/",
    epochs=20,
    batch_size=16,
    grad_accum_steps=1,
    output_dir = "/content/outputs"
)
```

Copy outputs to Drive:
```shell
!cp -r "/content/outputs" "/content/drive/MyDrive/uavs/"
```

If Colab restarts, restore from Drive first:
```shell
!cp -r /content/drive/MyDrive/uavs/outputs /content/outputs
```

Resuming training from a checkpoint:
```shell
from rfdetr import RFDETRNano

model = RFDETRNano()
model.train(
    dataset_dir="/content/datasets/",
    epochs=50,           # total epochs — not "50 more", but the new total
    batch_size=16,
    grad_accum_steps=1,
    output_dir="/content/outputs",
    resume="/content/outputs/checkpoint_best_total.pth"   # <-- resume here
)
```

Exporting the model to ONNX format
```shell
model = RFDETRNano(pretrain_weights="/content/outputs/checkpoint_best_total.pth")
model.export(
    format="onnx",
    output_dir="/content/drive/MyDrive/uavs/onnx_export"
)
```


## Training a YOLO26 Model in Google Colab Environment

Create the data.yaml file
```shell
train: /content/datasets/train/images
val: /content/datasets/val/images
test: /content/datasets/test/images

nc: 1
names: ['object']
```

Upload data.yaml file, and image/label tar files to Google Drive. Google Drive for Desktop makes this easy

Install YOLO26 via Ultralytics
```shell
%pip install ultralytics supervision roboflow -q
import ultralytics
ultralytics.checks()
```

Set up HOME
```shell
import os
HOME = os.getcwd()
print(HOME)
```

Mount to drive
```shell
# Allow access to personal google drive and add new folders

# Connect Google Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=True) # This will prompt for authorization.

# This will create the uavs files if they don't exist.
folders =  ["uavs-yolo26/"]
for folder in folders:
  path = "/content/drive/MyDrive/" + folder
  if not os.path.exists(path): # Create the folder if it does not exist
    os.mkdir(path)
```

Move in the compressed files and unpack them.
```shell
!mkdir {HOME}/datasets
%cd {HOME}/datasets
from google.colab import userdata
uavs_folder = "/content/drive/MyDrive/uavs-yolo26/"

!mkdir -p /content/datasets/savedir/
!cp -r "/content/drive/MyDrive/uavs-yolo26/images.tar.gz" "/content/datasets/savedir/"

!cp -r "/content/drive/MyDrive/uavs-yolo26/labels.tar.gz" "/content/datasets/savedir/"

!tar xf /content/datasets/savedir/images.tar.gz --directory /content/datasets/savedir/

!tar xf /content/datasets/savedir/labels.tar.gz --directory /content/datasets/savedir/
```

Move the data to the directory that YOLO26 expects
```shell
## make the directories that yolo26 expects
!mkdir /content/datasets/train/
!mkdir /content/datasets/train/images/
!mkdir /content/datasets/train/labels/
!mkdir /content/datasets/test/
!mkdir /content/datasets/test/images/
!mkdir /content/datasets/test/labels/
!mkdir /content/datasets/val/
!mkdir /content/datasets/val/images/
!mkdir /content/datasets/val/labels/

#get the data.yaml file
!cp "/content/drive/MyDrive/uavs-yolo26/data.yaml" "/content/datasets/data.yaml"
!ls /content/datasets/

#move the data to the expected directories
!cp -r "/content/datasets/savedir/images/train/" "/content/datasets/train/images/"
!cp -r "/content/datasets/savedir/labels/train/" "/content/datasets/train/labels/"

!cp -r "/content/datasets/savedir/images/test/" "/content/datasets/test/images/"
!cp -r "/content/datasets/savedir/labels/test/" "/content/datasets/test/labels/"

!cp -r "/content/datasets/savedir/images/val/" "/content/datasets/val/images/"
!cp -r "/content/datasets/savedir/labels/val/" "/content/datasets/val/labels/"

!ls /content/datasets/
```

Train a YOLO26 Medium model for 150 epochs:
```shell
!yolo task=detect mode=train model=yolo26m.pt data=/content/datasets/data.yaml \
  batch=128 epochs=150 patience=30 imgsz=640 \
  mixup=0.3 scale=0.9 plots=True
```

Save outputs to drive
```shell
!cp "/content/datasets/runs/detect/train/weights/best.pt" "/content/drive/MyDrive/uavs-yolo26/best.pt"
!cp "/content/datasets/runs/detect/train/weights/last.pt" "/content/drive/MyDrive/uavs-yolo26/last.pt"
!cp -r "/content/datasets/runs/detect/train/" "/content/drive/MyDrive/uavs-yolo26/train/"
```

You can train the model again for more epochs by loading the last checkpoint:
```shell
# copy last checkpoint from drive
!cp /content/drive/MyDrive/uavs-yolo26/last.pt /content/last.pt
```

Train for another 150 epochs starting from the last checkpoint:
```shell
!yolo task=detect mode=train model=/content/last.pt data=/content/datasets/data.yaml \
 batch=128 epochs=150 patience=30 imgsz=640 \
 mixup=0.3 scale=0.9 plots=True
```

Export model to ONNX format:
```shell
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/uavs-yolo26/best.pt")
model.export(format="onnx")
```