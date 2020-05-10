# I. Introduction

- YOLOv4 have been released on 23 Apr 2020. The new updated version of the popular YOLO object detection neural network which achieves state-of-the-art results on the MS-COCO dataset, running at real-time speed of more than 65 FPS.
- Darknet is a framework to train neural networks, it is open source and written in C/CUDA and serves as the basis for YOLO. The original repository, by J Redmon (also first author of the YOLO paper), can be found [here](https://github.com/pjreddie/darknet). Darknet is used as the framework for training YOLO, meaning it sets the architecture of the network.

     ![alt](https://imageai.readthedocs.io/en/latest/_images/image1.jpg)

- Open Images Dataset v4,provided by Google, is the largest existing dataset with object location annotations with ~9M images for 600 object classes that have been annotated with image-level labels and object bounding boxes.More details about OIDv4 can be read from [here](https://medium.com/@karol_majek/open-images-dataset-v4-237fc006b094).

- The OIDv4 Toolkit is  easy to use repo allows you to download only the image categories you need, along with their bounding boxes. This is public at https://github.com/EscVM/OIDv4_ToolKit
 
- The problem is that the difference between the label format in Open Images Dataset and the input format of Yolo model makes it a bit difficult and takes a long time to standardize.

- In this repository, I have customized and added a few features to easily standardize all data labels. It make easier to use custom data from Open Images Dataset for retraining the YOLO model.

- I used main code from https://github.com/AlexeyAB/darknet and https://github.com/EscVM/OIDv4_ToolKit.

- I also give a tutorial to train YOLOv4 on google colab with this repository and custom data.

# II. Training Yolo_v4 with custom data by Google colab
## 1. Setup google colab

- Check GPU
```
!nvidia-smi
```

    Sun May 10 07:04:58 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 440.82       Driver Version: 418.67       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   65C    P8    12W /  70W |      0MiB / 15079MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    

- Clone this repository

```
!git clone https://github.com/DucLeTrong/Train-YOLOv4-with-OpenImagesV4-Colab.git
```

## 2. Download data

### Install the required packages

Peek inside the requirements file if you have everything already installed. Most of the dependencies are common libraries.


```
%cd /content/Train-YOLOv4-with-OpenImagesV4-Colab/OIDv4_ToolKit
!pip3 install -r requirements.txt
```

### Download different classes in separated folders

 The ToolKit can be used to download classes in separated folders. The argument --classes accepts a list of classes or the path to the file.txt. 

In this example download Person and Car from the train set. In this case we have to use the following command.

- Read more about OIDv4_ToolKit  [here](https://github.com/DucLeTrong/darknet/tree/master/OIDv4_ToolKit)

```
 !python3 main.py downloader -y --classes Person Car --type_csv train --limit 100
```

    [92m
    		   ___   _____  ______            _    _    
    		 .'   `.|_   _||_   _ `.         | |  | |   
    		/  .-.  \ | |    | | `. \ _   __ | |__| |_  
    		| |   | | | |    | |  | |[ \ [  ]|____   _| 
    		\  `-'  /_| |_  _| |_.' / \ \/ /     _| |_  
    		 `.___.'|_____||______.'   \__/     |_____|
    	[0m
    [92m
                 _____                    _                 _             
                (____ \                  | |               | |            
                 _   \ \ ___  _ _ _ ____ | | ___   ____  _ | | ____  ____ 
                | |   | / _ \| | | |  _ \| |/ _ \ / _  |/ || |/ _  )/ ___)
                | |__/ / |_| | | | | | | | | |_| ( ( | ( (_| ( (/ /| |    
                |_____/ \___/ \____|_| |_|_|\___/ \_||_|\____|\____)_|    
                                                              
            [0m
        [INFO] | Downloading Person.[0m
    
    [95mPerson[0m
        [INFO] | Downloading train images.[0m
        [INFO] | [INFO] Found 248384 online images for train.[0m
        [INFO] | Limiting to 100 images.[0m
        [INFO] | Download of 100 images in train.[0m
    100% 100/100 [01:02<00:00,  1.59it/s]
        [INFO] | Done![0m
        [INFO] | Creating labels for Person of train.[0m
        [INFO] | Labels creation completed.[0m
        [INFO] | Downloading Car.[0m
    
    [95mCar[0m
        [INFO] | Downloading train images.[0m
        [INFO] | [INFO] Found 89465 online images for train.[0m
        [INFO] | Limiting to 100 images.[0m
        [INFO] | Download of 100 images in train.[0m
    100% 100/100 [00:57<00:00,  1.73it/s]
        [INFO] | Done![0m
        [INFO] | Creating labels for Car of train.[0m
        [INFO] | Labels creation completed.[0m
    

## 3. Prepare data for Yolo formatting

### Making obj_name.txt file contain all classes


```
%cd /content/Train-YOLOv4-with-OpenImagesV4-Colab

!echo Person >> obj_name.txt
!echo Car >> obj_name.txt
```

    /content/Train-YOLOv4-with-OpenImagesV4-Colab
    

### Processing data to yolo formatting and preparing training data with  process_data.py


```
!python3 process_data.py --data_set_name='Train' --des_path='custom_data'
```

## 4. Config model

### Mount google drive and create a symbolic link backup file at google drive to save training weight in case of training interruption with internet problems. 


```
from google.colab import drive
drive.mount('/content/drive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive
    


```
!ln -s "/content/drive/My Drive/backup_yolo" "/content/Train-YOLOv4-with-OpenImagesV4-Colab"
```

### I created a text file named it 'config_data.txt' containing the following:

- classes is '2' as number of classes training.

- train.txt is a single text file that lists the full directory of where each photo is that you downloaded; I created it by runing process_data.py

- test.txt; same as above and this is created along with the train.txt file.

- "backup_yolo" is just a folder where you want the trained weights files to be output to while it trains.


```
# Create config_data.txt config file
!rm -rf config_data.txt
!echo classes=2 > config_data.txt
!echo train=train.txt >> config_data.txt
!echo valid=test.txt >> config_data.txt
!echo names=obj_name.txt >> config_data.txt
!echo backup=backup_yolo >> config_data.txt
```

### Compile Darknet (using cmake)



```
!chmod +x ./build.sh
!./build.sh

```
    [100%] [32m[1mLinking CXX executable uselib[0m
    [100%] Built target uselib
    [36mInstall the project...[0m
    -- Install configuration: "Release"
    -- Installing: /content/Train-YOLOv4-with-OpenImagesV4-Colab/libdark.so
    -- Installing: /content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet/darknet.h
    -- Installing: /content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet/yolo_v2_class.hpp
    -- Installing: /content/Train-YOLOv4-with-OpenImagesV4-Colab/uselib
    -- Set runtime path of "/content/Train-YOLOv4-with-OpenImagesV4-Colab/uselib" to ""
    -- Installing: /content/Train-YOLOv4-with-OpenImagesV4-Colab/darknet
    -- Installing: /content/Train-YOLOv4-with-OpenImagesV4-Colab/share/darknet/DarknetTargets.cmake
    -- Installing: /content/Train-YOLOv4-with-OpenImagesV4-Colab/share/darknet/DarknetTargets-release.cmake
    -- Installing: /content/Train-YOLOv4-with-OpenImagesV4-Colab/share/darknet/DarknetConfig.cmake
    -- Installing: /content/Train-YOLOv4-with-OpenImagesV4-Colab/share/darknet/DarknetConfigVersion.cmake
    

### Download weights of YoloV4 pretrain from google drive


```
import gdown
url = 'https://drive.google.com/uc?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp&export=download'
output = 'yolov4.conv.137'
gdown.download(url, output, quiet=False) 
```
    Downloading...
    From: https://drive.google.com/uc?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp&export=download
    To: /content/Train-YOLOv4-with-OpenImagesV4-Colab/yolov4.conv.137
    170MB [00:03, 44.3MB/s]
    
    'yolov4.conv.137'



### Customize model config file

#### Copy config file


```
!cp cfg/yolov4-custom.cfg yolov4-custom.txt
```

#### Edit yolov4-custom.txt file as follows: 

- Line 3: Set batch=64.

- Line 4: Set subdivisions=32, the batch will be divided by 16 or 64 depends on GPU VRAM requirements.

- Change line max_batches to classes*2000 but not less than number of training images, and not less than 6000, f.e. max_batches=6000 if you train for 3 classes.

- Change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400.

- Change line classes=2 to your number of objects in each of 3: Line 970 1058 1146.

- Change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers. Line 1139 1051 963. In this case, filters=(2 + 5)x3

#### Save that after making the changes.

## 5. Train Model


```
!chmod +x ./darknet
!./darknet detector train config_data.txt yolov4-custom.txt yolov4.conv.137 -dont_show > yolov3-5c.log
```

     CUDA-version: 10010 (10010), cuDNN: 7.6.5, CUDNN_HALF=1, GPU count: 1  
     OpenCV version: 3.2.0
     0 : compute_capability = 750, cudnn_half = 1, GPU: Tesla T4 
       layer   filters  size/strd(dil)      input                output
       0 conv     32       3 x 3/ 1    608 x 608 x   3 ->  608 x 608 x  32 0.639 BF
       1 conv     64       3 x 3/ 2    608 x 608 x  32 ->  304 x 304 x  64 3.407 BF
       2 conv     64       1 x 1/ 1    304 x 304 x  64 ->  304 x 304 x  64 0.757 BF
       3 route  1 		                           ->  304 x 304 x  64 
       4 conv     64       1 x 1/ 1    304 x 304 x  64 ->  304 x 304 x  64 0.757 BF
       5 conv     32       1 x 1/ 1    304 x 304 x  64 ->  304 x 304 x  32 0.379 BF
       6 conv     64       3 x 3/ 1    304 x 304 x  32 ->  304 x 304 x  64 3.407 BF
       7 Shortcut Layer: 4,  wt = 0, wn = 0, outputs: 304 x 304 x  64 0.006 BF
       8 conv     64       1 x 1/ 1    304 x 304 x  64 ->  304 x 304 x  64 0.757 BF
       9 route  8 2 	                           ->  304 x 304 x 128

# III. Reference
 ### 1. https://arxiv.org/abs/2004.10934
 ### 2. https://github.com/AlexeyAB/darknet
 ### 3. https://github.com/EscVM/OIDv4_ToolKit
 ### 4. https://phamdinhkhanh.github.io/2020/03/10/DarknetGoogleColab.html 
