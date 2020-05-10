# 1. Setup google colab


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
    

Clone this repository


```
!git clone https://github.com/DucLeTrong/Train-YOLOv4-with-OpenImagesV4-Colab.git
```

#2. Download data

Install the required packages

Peek inside the requirements file if you have everything already installed. Most of the dependencies are common libraries.


```
%cd /content/Train-YOLOv4-with-OpenImagesV4-Colab/OIDv4_ToolKit
!pip3 install -r requirements.txt
```

 Download different classes in separated folders

 The ToolKit can be used to download classes in separated folders. The argument --classes accepts a list of classes or the path to the file.txt. 

In this example download Person and Car from the train set. In this case we have to use the following command.


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
    

#3. Prepare data for Yolo formatting

Making obj_name.txt file contain all classes


```
%cd /content/Train-YOLOv4-with-OpenImagesV4-Colab

!echo Person >> obj_name.txt
!echo Car >> obj_name.txt
```

    /content/Train-YOLOv4-with-OpenImagesV4-Colab
    

Processing data to yolo formatting and preparing training data with  process_data.py


```
!python3 process_data.py --data_set_name='Train' --des_path='custom_data'
```

#4. Config model

Mount google drive and create a symbolic link backup file at google drive to save training weight in case of training interruption with internet problems. 


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

 I created a text file named it 'config_data.txt' containing the following:
 -classes is '2' as number of classes training.

-train.txt is a single text file that lists the full directory of where each photo is that you downloaded; I created it by runing process_data.py

-test.txt; same as above and this is created along with the train.txt file.

-"backup" is just a folder where you want the trained weights files to be output to while it trains.


```
# Create config_data.txt config file
!!rm -rf config_data.txt
!echo classes=2 > config_data.txt
!echo train=train.txt >> config_data.txt
!echo valid=test.txt >> config_data.txt
!echo names=obj_name.txt >> config_data.txt
!echo backup=backup_yolo >> config_data.txt
```

Compile Darknet (using cmake)



```
!chmod +x ./build.sh
!./build.sh

```

    -- The C compiler identification is GNU 7.5.0
    -- The CXX compiler identification is GNU 7.5.0
    -- Check for working C compiler: /usr/bin/cc
    -- Check for working C compiler: /usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /usr/bin/c++
    -- Check for working CXX compiler: /usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Looking for a CUDA compiler
    -- Looking for a CUDA compiler - /usr/local/cuda/bin/nvcc
    -- The CUDA compiler identification is NVIDIA 10.1.243
    -- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc
    -- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc -- works
    -- Detecting CUDA compiler ABI info
    -- Detecting CUDA compiler ABI info - done
    -- Looking for pthread.h
    -- Looking for pthread.h - found
    -- Looking for pthread_create
    -- Looking for pthread_create - not found
    -- Looking for pthread_create in pthreads
    -- Looking for pthread_create in pthreads - not found
    -- Looking for pthread_create in pthread
    -- Looking for pthread_create in pthread - found
    -- Found Threads: TRUE  
    -- Found CUDA: /usr/local/cuda (found version "10.1") 
    -- Autodetected CUDA architecture(s):  7.5
    -- Building with CUDA flags: -gencode;arch=compute_75,code=sm_75
    -- Your setup supports half precision (it requires CC >= 7.0)
    -- Found OpenCV: /usr (found version "3.2.0") 
    -- Found Stb: /content/Train-YOLOv4-with-OpenImagesV4-Colab/3rdparty/stb/include  
    -- Found OpenMP_C: -fopenmp (found version "4.5") 
    -- Found OpenMP_CXX: -fopenmp (found version "4.5") 
    -- Found OpenMP: TRUE (found version "4.5")  
    --   ->  darknet is fine for now, but uselib_track has been disabled!
    --   ->  Please rebuild OpenCV from sources with CUDA support to enable it
    -- Found CUDNN: /usr/include (found version "7.6.5") 
    -- CMAKE_CUDA_FLAGS: -gencode arch=compute_75,code=sm_75 --compiler-options " -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings -DGPU -DCUDNN -DOPENCV -fPIC -fopenmp -Ofast " 
    -- ZED SDK not found
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /content/Train-YOLOv4-with-OpenImagesV4-Colab/build_release
    [35m[1mScanning dependencies of target darknet[0m
    [35m[1mScanning dependencies of target dark[0m
    [  1%] [32mBuilding C object CMakeFiles/darknet.dir/src/batchnorm_layer.c.o[0m
    [  2%] [32mBuilding C object CMakeFiles/darknet.dir/src/activations.c.o[0m
    [  2%] [32mBuilding C object CMakeFiles/darknet.dir/src/blas.c.o[0m
    [  2%] [32mBuilding C object CMakeFiles/darknet.dir/src/activation_layer.c.o[0m
    [  3%] [32mBuilding C object CMakeFiles/darknet.dir/src/darknet.c.o[0m
    [  4%] [32mBuilding C object CMakeFiles/darknet.dir/src/avgpool_layer.c.o[0m
    [  4%] [32mBuilding C object CMakeFiles/darknet.dir/src/art.c.o[0m
    [  4%] [32mBuilding CXX object CMakeFiles/dark.dir/src/yolo_v2_class.cpp.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:[m[K In function ‘[01m[Kactivate[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KRELU6[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
         [01;35m[Kswitch[m[K(a){
         [01;35m[K^~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KSWISH[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KMISH[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KNORM_CHAN[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KNORM_CHAN_SOFTMAX[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KNORM_CHAN_SOFTMAX_MAXVAL[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:[m[K In function ‘[01m[Kgradient[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas.c:[m[K In function ‘[01m[Kbackward_shortcut_multilayer_cpu[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:287:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KSWISH[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
         [01;35m[Kswitch[m[K(a){
         [01;35m[K^~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:287:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KMISH[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas.c:207:21:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kout_index[m[K’ [[01;35m[K-Wunused-variable[m[K]
                     int [01;35m[Kout_index[m[K = id;
                         [01;35m[K^~~~~~~~~[m[K
    [  5%] [32mBuilding C object CMakeFiles/darknet.dir/src/box.c.o[0m
    [  5%] [32mBuilding C object CMakeFiles/darknet.dir/src/captcha.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/box.c:[m[K In function ‘[01m[Kbox_iou_kind[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/box.c:154:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KMSE[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
         [01;35m[Kswitch[m[K(iou_kind) {
         [01;35m[K^~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/box.c:[m[K In function ‘[01m[Kdiounms_sort[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/box.c:898:27:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kbeta_prob[m[K’ [[01;35m[K-Wunused-variable[m[K]
                         float [01;35m[Kbeta_prob[m[K = pow(dets[j].prob[k], 2) / sum_prob;
                               [01;35m[K^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/box.c:897:27:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kalpha_prob[m[K’ [[01;35m[K-Wunused-variable[m[K]
                         float [01;35m[Kalpha_prob[m[K = pow(dets[i].prob[k], 2) / sum_prob;
                               [01;35m[K^~~~~~~~~~[m[K
    [  6%] [32mBuilding C object CMakeFiles/darknet.dir/src/cifar.c.o[0m
    [  7%] [32mBuilding C object CMakeFiles/darknet.dir/src/classifier.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:[m[K In function ‘[01m[Ktrain_classifier[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:138:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kcount[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kcount[m[K = 0;
             [01;35m[K^~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:[m[K In function ‘[01m[Kpredict_classifier[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:831:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Ktime[m[K’ [[01;35m[K-Wunused-variable[m[K]
         clock_t [01;35m[Ktime[m[K;
                 [01;35m[K^~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:[m[K In function ‘[01m[Kdemo_classifier[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:1261:49:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Ktval_result[m[K’ [[01;35m[K-Wunused-variable[m[K]
             struct timeval tval_before, tval_after, [01;35m[Ktval_result[m[K;
                                                     [01;35m[K^~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:1261:37:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Ktval_after[m[K’ [[01;35m[K-Wunused-variable[m[K]
             struct timeval tval_before, [01;35m[Ktval_after[m[K, tval_result;
                                         [01;35m[K^~~~~~~~~~[m[K
    [  7%] [32mBuilding C object CMakeFiles/darknet.dir/src/coco.c.o[0m
    [  8%] [32mBuilding C object CMakeFiles/darknet.dir/src/col2im.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/coco.c:[m[K In function ‘[01m[Kvalidate_coco_recall[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/coco.c:248:11:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kbase[m[K’ [[01;35m[K-Wunused-variable[m[K]
         char *[01;35m[Kbase[m[K = "results/comp4_det_test_";
               [01;35m[K^~~~[m[K
    [  9%] [32mBuilding C object CMakeFiles/dark.dir/src/activation_layer.c.o[0m
    [  9%] [32mBuilding C object CMakeFiles/dark.dir/src/activations.c.o[0m
    [  9%] [32mBuilding C object CMakeFiles/darknet.dir/src/compare.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:[m[K In function ‘[01m[Kactivate[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KRELU6[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
         [01;35m[Kswitch[m[K(a){
         [01;35m[K^~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KSWISH[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KMISH[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KNORM_CHAN[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KNORM_CHAN_SOFTMAX[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:77:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KNORM_CHAN_SOFTMAX_MAXVAL[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:[m[K In function ‘[01m[Kgradient[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:287:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KSWISH[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
         [01;35m[Kswitch[m[K(a){
         [01;35m[K^~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.c:287:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KMISH[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
    [ 10%] [32mBuilding C object CMakeFiles/dark.dir/src/art.c.o[0m
    [ 11%] [32mBuilding C object CMakeFiles/darknet.dir/src/connected_layer.c.o[0m
    [ 12%] [32mBuilding C object CMakeFiles/darknet.dir/src/conv_lstm_layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:[m[K In function ‘[01m[Kforward_connected_layer_gpu[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:346:11:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kone[m[K’ [[01;35m[K-Wunused-variable[m[K]
         float [01;35m[Kone[m[K = 1;    // alpha[0], beta[0]
               [01;35m[K^~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:344:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kc[m[K’ [[01;35m[K-Wunused-variable[m[K]
         float * [01;35m[Kc[m[K = l.output_gpu;
                 [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:343:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kb[m[K’ [[01;35m[K-Wunused-variable[m[K]
         float * [01;35m[Kb[m[K = l.weights_gpu;
                 [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:342:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Ka[m[K’ [[01;35m[K-Wunused-variable[m[K]
         float * [01;35m[Ka[m[K = state.input;
                 [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:341:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kn[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kn[m[K = l.outputs;
             [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:340:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kk[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kk[m[K = l.inputs;
             [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:339:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Km[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Km[m[K = l.batch;
             [01;35m[K^[m[K
    [ 13%] [32mBuilding C object CMakeFiles/dark.dir/src/avgpool_layer.c.o[0m
    [ 13%] [32mBuilding C object CMakeFiles/darknet.dir/src/convolutional_layer.c.o[0m
    [ 14%] [32mBuilding C object CMakeFiles/darknet.dir/src/cost_layer.c.o[0m
    [ 14%] [32mBuilding C object CMakeFiles/darknet.dir/src/cpu_gemm.c.o[0m
    [ 15%] [32mBuilding C object CMakeFiles/darknet.dir/src/crnn_layer.c.o[0m
    [ 16%] [32mBuilding C object CMakeFiles/darknet.dir/src/crop_layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/convolutional_layer.c:[m[K In function ‘[01m[Kforward_convolutional_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/convolutional_layer.c:1204:32:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kt_intput_size[m[K’ [[01;35m[K-Wunused-variable[m[K]
                             size_t [01;35m[Kt_intput_size[m[K = binary_transpose_align_input(k, n, state.workspace, &l.t_bit_input, ldb_align, l.bit_align);
                                    [01;35m[K^~~~~~~~~~~~~[m[K
    [ 16%] [32mBuilding C object CMakeFiles/dark.dir/src/batchnorm_layer.c.o[0m
    [ 16%] [32mBuilding C object CMakeFiles/darknet.dir/src/dark_cuda.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:[m[K In function ‘[01m[Kcudnn_check_error_extended[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:224:20:[m[K [01;35m[Kwarning: [m[Kcomparison between ‘[01m[KcudaError_t {aka enum cudaError}[m[K’ and ‘[01m[Kenum <anonymous>[m[K’ [[01;35m[K-Wenum-compare[m[K]
             if (status [01;35m[K!=[m[K CUDNN_STATUS_SUCCESS)
                        [01;35m[K^~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:[m[K In function ‘[01m[Kpre_allocate_pinned_memory[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:276:40:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%u[m[K’ expects argument of type ‘[01m[Kunsigned int[m[K’, but argument 2 has type ‘[01m[Klong unsigned int[m[K’ [[01;35m[K-Wformat=[m[K]
             printf("pre_allocate: size = [01;35m[K%Iu[m[K MB, num_of_blocks = %Iu, block_size = %Iu MB \n",
                                          [01;35m[K~~^[m[K
                                          [32m[K%Ilu[m[K
                 [32m[Ksize / (1024*1024)[m[K, num_of_blocks, pinned_block_size / (1024 * 1024));
                 [32m[K~~~~~~~~~~~~~~~~~~[m[K          
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:276:64:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%u[m[K’ expects argument of type ‘[01m[Kunsigned int[m[K’, but argument 3 has type ‘[01m[Ksize_t {aka const long unsigned int}[m[K’ [[01;35m[K-Wformat=[m[K]
             printf("pre_allocate: size = %Iu MB, num_of_blocks = [01;35m[K%Iu[m[K, block_size = %Iu MB \n",
                                                                  [01;35m[K~~^[m[K
                                                                  [32m[K%Ilu[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:276:82:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%u[m[K’ expects argument of type ‘[01m[Kunsigned int[m[K’, but argument 4 has type ‘[01m[Klong unsigned int[m[K’ [[01;35m[K-Wformat=[m[K]
             printf("pre_allocate: size = %Iu MB, num_of_blocks = %Iu, block_size = [01;35m[K%Iu[m[K MB \n",
                                                                                    [01;35m[K~~^[m[K
                                                                                    [32m[K%Ilu[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:286:37:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%d[m[K’ expects argument of type ‘[01m[Kint[m[K’, but argument 2 has type ‘[01m[Ksize_t {aka const long unsigned int}[m[K’ [[01;35m[K-Wformat=[m[K]
                     printf(" Allocated [01;35m[K%d[m[K pinned block \n", pinned_block_size);
                                        [01;35m[K~^[m[K
                                        [32m[K%ld[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:[m[K In function ‘[01m[Kcuda_make_array_pinned_preallocated[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:307:43:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%d[m[K’ expects argument of type ‘[01m[Kint[m[K’, but argument 2 has type ‘[01m[Ksize_t {aka long unsigned int}[m[K’ [[01;35m[K-Wformat=[m[K]
                 printf("\n Pinned block_id = [01;35m[K%d[m[K, filled = %f %% \n", pinned_block_id, filled);
                                              [01;35m[K~^[m[K
                                              [32m[K%ld[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:322:64:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%d[m[K’ expects argument of type ‘[01m[Kint[m[K’, but argument 2 has type ‘[01m[Klong unsigned int[m[K’ [[01;35m[K-Wformat=[m[K]
                 printf("Try to allocate new pinned memory, size = [01;35m[K%d[m[K MB \n", [32m[Ksize / (1024 * 1024)[m[K);
                                                                   [01;35m[K~^[m[K         [32m[K~~~~~~~~~~~~~~~~~~~~[m[K
                                                                   [32m[K%ld[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:328:63:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%d[m[K’ expects argument of type ‘[01m[Kint[m[K’, but argument 2 has type ‘[01m[Klong unsigned int[m[K’ [[01;35m[K-Wformat=[m[K]
                 printf("Try to allocate new pinned BLOCK, size = [01;35m[K%d[m[K MB \n", [32m[Ksize / (1024 * 1024)[m[K);
                                                                  [01;35m[K~^[m[K         [32m[K~~~~~~~~~~~~~~~~~~~~[m[K
                                                                  [32m[K%ld[m[K
    [ 17%] [32mBuilding C object CMakeFiles/darknet.dir/src/data.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/data.c:[m[K In function ‘[01m[Kload_data_detection[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/data.c:1238:24:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kx[m[K’ [[01;35m[K-Wunused-variable[m[K]
                     int k, [01;35m[Kx[m[K, y;
                            [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/data.c:1038:43:[m[K [01;35m[Kwarning: [m[Kvariable ‘[01m[Kr_scale[m[K’ set but not used [[01;35m[K-Wunused-but-set-variable[m[K]
         float r1 = 0, r2 = 0, r3 = 0, r4 = 0, [01;35m[Kr_scale[m[K = 0;
                                               [01;35m[K^~~~~~~[m[K
    [ 18%] [32mBuilding C object CMakeFiles/dark.dir/src/blas.c.o[0m
    [ 18%] [32mBuilding C object CMakeFiles/dark.dir/src/box.c.o[0m
    [ 19%] [32mBuilding C object CMakeFiles/dark.dir/src/captcha.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas.c:[m[K In function ‘[01m[Kbackward_shortcut_multilayer_cpu[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas.c:207:21:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kout_index[m[K’ [[01;35m[K-Wunused-variable[m[K]
                     int [01;35m[Kout_index[m[K = id;
                         [01;35m[K^~~~~~~~~[m[K
    [ 19%] [32mBuilding C object CMakeFiles/darknet.dir/src/deconvolutional_layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/box.c:[m[K In function ‘[01m[Kbox_iou_kind[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/box.c:154:5:[m[K [01;35m[Kwarning: [m[Kenumeration value ‘[01m[KMSE[m[K’ not handled in switch [[01;35m[K-Wswitch[m[K]
         [01;35m[Kswitch[m[K(iou_kind) {
         [01;35m[K^~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/box.c:[m[K In function ‘[01m[Kdiounms_sort[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/box.c:898:27:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kbeta_prob[m[K’ [[01;35m[K-Wunused-variable[m[K]
                         float [01;35m[Kbeta_prob[m[K = pow(dets[j].prob[k], 2) / sum_prob;
                               [01;35m[K^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/box.c:897:27:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kalpha_prob[m[K’ [[01;35m[K-Wunused-variable[m[K]
                         float [01;35m[Kalpha_prob[m[K = pow(dets[i].prob[k], 2) / sum_prob;
                               [01;35m[K^~~~~~~~~~[m[K
    [ 20%] [32mBuilding C object CMakeFiles/dark.dir/src/cifar.c.o[0m
    [ 21%] [32mBuilding C object CMakeFiles/darknet.dir/src/demo.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/demo.c:[m[K In function ‘[01m[Kdetect_in_thread[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/demo.c:102:16:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kprediction[m[K’ [[01;35m[K-Wunused-variable[m[K]
             float *[01;35m[Kprediction[m[K = network_predict(net, X);
                    [01;35m[K^~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/demo.c:100:15:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kl[m[K’ [[01;35m[K-Wunused-variable[m[K]
             layer [01;35m[Kl[m[K = net.layers[net.n - 1];
                   [01;35m[K^[m[K
    [ 21%] [32mBuilding C object CMakeFiles/dark.dir/src/classifier.c.o[0m
    [ 22%] [32mBuilding C object CMakeFiles/darknet.dir/src/detection_layer.c.o[0m
    In file included from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_v2_class.cpp:2:0[m[K:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:[m[K In constructor ‘[01m[Ktrack_kalman_t::track_kalman_t(int, int, float, cv::Size)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:708:14:[m[K [01;35m[Kwarning: [m[K‘[01m[Ktrack_kalman_t::img_size[m[K’ will be initialized after [[01;35m[K-Wreorder[m[K]
         cv::Size [01;35m[Kimg_size[m[K;  // max value of x,y,w,h
                  [01;35m[K^~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:700:9:[m[K [01;35m[Kwarning: [m[K  ‘[01m[Kint track_kalman_t::track_id_counter[m[K’ [[01;35m[K-Wreorder[m[K]
         int [01;35m[Ktrack_id_counter[m[K;
             [01;35m[K^~~~~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:853:5:[m[K [01;35m[Kwarning: [m[K  when initialized here [[01;35m[K-Wreorder[m[K]
         [01;35m[Ktrack_kalman_t[m[K(int _max_objects = 1000, int _min_frames = 3, float _max_dist = 40, cv::Size _img_size = cv::Size(10000, 10000)) :
         [01;35m[K^~~~~~~~~~~~~~[m[K
    [ 22%] [32mBuilding C object CMakeFiles/darknet.dir/src/detector.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:[m[K In member function ‘[01m[Kvoid track_kalman_t::clear_old_states()[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:879:50:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
                     if ((result_vec_pred[state_id].x > img_size.width) ||
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:880:50:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
                         (result_vec_pred[state_id].y > img_size.height))
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:[m[K In member function ‘[01m[Ktrack_kalman_t::tst_t track_kalman_t::get_state_id(bbox_t, std::vector<bool>&)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:900:30:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
             for (size_t i = 0; [01;35m[Ki < max_objects[m[K; ++i)
                                [01;35m[K~~^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:[m[K In member function ‘[01m[Kstd::vector<bbox_t> track_kalman_t::predict()[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:990:30:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
             for (size_t i = 0; [01;35m[Ki < max_objects[m[K; ++i)
                                [01;35m[K~~^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:[m[K In member function ‘[01m[Kstd::vector<bbox_t> track_kalman_t::correct(std::vector<bbox_t>)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:1025:30:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
             for (size_t i = 0; [01;35m[Ki < max_objects[m[K; ++i)
                                [01;35m[K~~^~~~~~~~~~~~~[m[K
    [ 23%] [32mBuilding C object CMakeFiles/dark.dir/src/coco.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:[m[K In function ‘[01m[Ktrain_classifier[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:138:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kcount[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kcount[m[K = 0;
             [01;35m[K^~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:[m[K In function ‘[01m[Kpredict_classifier[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:831:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Ktime[m[K’ [[01;35m[K-Wunused-variable[m[K]
         clock_t [01;35m[Ktime[m[K;
                 [01;35m[K^~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:[m[K In function ‘[01m[Kdemo_classifier[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:1261:49:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Ktval_result[m[K’ [[01;35m[K-Wunused-variable[m[K]
             struct timeval tval_before, tval_after, [01;35m[Ktval_result[m[K;
                                                     [01;35m[K^~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/classifier.c:1261:37:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Ktval_after[m[K’ [[01;35m[K-Wunused-variable[m[K]
             struct timeval tval_before, [01;35m[Ktval_after[m[K, tval_result;
                                         [01;35m[K^~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_v2_class.cpp:[m[K In member function ‘[01m[Kstd::vector<bbox_t> Detector::tracking_id(std::vector<bbox_t>, bool, int, int)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_v2_class.cpp:370:40:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
             if ([01;35m[Kprev_bbox_vec_deque.size() > frames_story[m[K) prev_bbox_vec_deque.pop_back();
                 [01;35m[K~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_v2_class.cpp:385:34:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
                         if ([01;35m[Kcur_dist < max_dist[m[K && (k.track_id == 0 || dist_vec[m] > cur_dist)) {
                             [01;35m[K~~~~~~~~~^~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_v2_class.cpp:409:40:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
             if ([01;35m[Kprev_bbox_vec_deque.size() > frames_story[m[K) prev_bbox_vec_deque.pop_back();
                 [01;35m[K~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:[m[K In function ‘[01m[Kprint_cocos[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:453:29:[m[K [01;35m[Kwarning: [m[Kformat not a string literal and no format arguments [[01;35m[K-Wformat-security[m[K]
                     fprintf(fp, [01;35m[Kbuff[m[K);
                                 [01;35m[K^~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/coco.c:[m[K In function ‘[01m[Kvalidate_coco_recall[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:[m[K In function ‘[01m[Keliminate_bdd[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:546:21:[m[K [01;35m[Kwarning: [m[Kstatement with no effect [[01;35m[K-Wunused-value[m[K]
                         [01;35m[Kfor[m[K (k; buf[k + n] != '\0'; k++)
                         [01;35m[K^~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:[m[K In function ‘[01m[Kvalidate_detector[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/coco.c:248:11:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kbase[m[K’ [[01;35m[K-Wunused-variable[m[K]
         char *[01;35m[Kbase[m[K = "results/comp4_det_test_";
               [01;35m[K^~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:659:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kmkd2[m[K’ [[01;35m[K-Wunused-variable[m[K]
             int [01;35m[Kmkd2[m[K = make_directory(buff2, 0777);
                 [01;35m[K^~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:657:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kmkd[m[K’ [[01;35m[K-Wunused-variable[m[K]
             int [01;35m[Kmkd[m[K = make_directory(buff, 0777);
                 [01;35m[K^~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:[m[K In function ‘[01m[Kvalidate_detector_map[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:1278:15:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kclass_recall[m[K’ [[01;35m[K-Wunused-variable[m[K]
             float [01;35m[Kclass_recall[m[K = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)(truth_classes_count[i] - tp_for_thresh_per_class[i]));
                   [01;35m[K^~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:1277:15:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kclass_precision[m[K’ [[01;35m[K-Wunused-variable[m[K]
             float [01;35m[Kclass_precision[m[K = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)fp_for_thresh_per_class[i]);
                   [01;35m[K^~~~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:[m[K In function ‘[01m[Kdraw_object[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:1794:19:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kinv_loss[m[K’ [[01;35m[K-Wunused-variable[m[K]
                 float [01;35m[Kinv_loss[m[K = 1.0 / max_val_cmp(0.01, avg_loss);
                       [01;35m[K^~~~~~~~[m[K
    [ 23%] [32mBuilding C object CMakeFiles/dark.dir/src/col2im.c.o[0m
    [ 24%] [32mBuilding C object CMakeFiles/darknet.dir/src/dice.c.o[0m
    [ 24%] [32mBuilding C object CMakeFiles/darknet.dir/src/dropout_layer.c.o[0m
    [ 25%] [32mBuilding C object CMakeFiles/dark.dir/src/compare.c.o[0m
    [ 26%] [32mBuilding C object CMakeFiles/dark.dir/src/connected_layer.c.o[0m
    [ 27%] [32mBuilding C object CMakeFiles/darknet.dir/src/gaussian_yolo_layer.c.o[0m
    [ 28%] [32mBuilding C object CMakeFiles/darknet.dir/src/gemm.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:[m[K In function ‘[01m[Kforward_connected_layer_gpu[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:346:11:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kone[m[K’ [[01;35m[K-Wunused-variable[m[K]
         float [01;35m[Kone[m[K = 1;    // alpha[0], beta[0]
               [01;35m[K^~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:344:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kc[m[K’ [[01;35m[K-Wunused-variable[m[K]
         float * [01;35m[Kc[m[K = l.output_gpu;
                 [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:343:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kb[m[K’ [[01;35m[K-Wunused-variable[m[K]
         float * [01;35m[Kb[m[K = l.weights_gpu;
                 [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:342:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Ka[m[K’ [[01;35m[K-Wunused-variable[m[K]
         float * [01;35m[Ka[m[K = state.input;
                 [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:341:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kn[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kn[m[K = l.outputs;
             [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:340:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kk[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kk[m[K = l.inputs;
             [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/connected_layer.c:339:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Km[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Km[m[K = l.batch;
             [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:[m[K In function ‘[01m[Kmake_gaussian_yolo_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:71:38:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
         if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Kl.output, batch*l.outputs * sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
                                          [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:7[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:78:38:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
         if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Kl.delta, batch*l.outputs * sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
                                          [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:7[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:[m[K In function ‘[01m[Kresize_gaussian_yolo_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:110:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess != cudaHostAlloc([01;35m[K&[m[Kl->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:7[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:119:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess != cudaHostAlloc([01;35m[K&[m[Kl->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:7[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [ 28%] [32mBuilding C object CMakeFiles/dark.dir/src/conv_lstm_layer.c.o[0m
    [ 28%] [32mBuilding C object CMakeFiles/darknet.dir/src/go.c.o[0m
    [ 29%] [32mBuilding C object CMakeFiles/dark.dir/src/convolutional_layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:[m[K In function ‘[01m[K_castu32_f32[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:550:5:[m[K [01;35m[Kwarning: [m[Kdereferencing type-punned pointer will break strict-aliasing rules [[01;35m[K-Wstrict-aliasing[m[K]
         [01;35m[Kreturn[m[K *((float *)&a);
         [01;35m[K^~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:[m[K In function ‘[01m[Kconvolution_2d[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1066:27:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Ksum[m[K’ [[01;35m[K-Wunused-variable[m[K]
                         float [01;35m[Ksum[m[K = 0;
                               [01;35m[K^~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1046:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kall256i_one[m[K’ [[01;35m[K-Wunused-variable[m[K]
         __m256i [01;35m[Kall256i_one[m[K = _mm256_set1_epi32(1);
                 [01;35m[K^~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1045:12:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kall256_one[m[K’ [[01;35m[K-Wunused-variable[m[K]
         __m256 [01;35m[Kall256_one[m[K = _mm256_set1_ps(1);
                [01;35m[K^~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1043:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kidx256[m[K’ [[01;35m[K-Wunused-variable[m[K]
         __m256i [01;35m[Kidx256[m[K = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
                 [01;35m[K^~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1040:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kall256_last_zero[m[K’ [[01;35m[K-Wunused-variable[m[K]
         __m256i [01;35m[Kall256_last_zero[m[K =
                 [01;35m[K^~~~~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:[m[K In function ‘[01m[Kim2col_cpu_custom_bin[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1644:17:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kall256_sing1[m[K’ [[01;35m[K-Wunused-variable[m[K]
             __m256i [01;35m[Kall256_sing1[m[K = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
                     [01;35m[K^~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/convolutional_layer.c:[m[K In function ‘[01m[Kforward_convolutional_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/convolutional_layer.c:1204:32:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kt_intput_size[m[K’ [[01;35m[K-Wunused-variable[m[K]
                             size_t [01;35m[Kt_intput_size[m[K = binary_transpose_align_input(k, n, state.workspace, &l.t_bit_input, ldb_align, l.bit_align);
                                    [01;35m[K^~~~~~~~~~~~~[m[K
    [ 30%] [32mBuilding C object CMakeFiles/darknet.dir/src/gru_layer.c.o[0m
    [ 30%] [32mBuilding C object CMakeFiles/darknet.dir/src/im2col.c.o[0m
    [ 30%] [32mBuilding C object CMakeFiles/dark.dir/src/cost_layer.c.o[0m
    [ 31%] [32mBuilding C object CMakeFiles/darknet.dir/src/image.c.o[0m
    [ 32%] [32mBuilding C object CMakeFiles/dark.dir/src/cpu_gemm.c.o[0m
    [ 33%] [32mBuilding C object CMakeFiles/dark.dir/src/crnn_layer.c.o[0m
    [ 34%] [32mBuilding C object CMakeFiles/darknet.dir/src/layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.c:[m[K In function ‘[01m[Kfree_layer_custom[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.c:191:9:[m[K [01;35m[Kwarning: [m[Kthis ‘[01m[Kif[m[K’ clause does not guard... [[01;35m[K-Wmisleading-indentation[m[K]
             [01;35m[Kif[m[K (l.x_gpu)                   cuda_free(l.x_gpu);  l.x_gpu = NULL;
             [01;35m[K^~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.c:191:61:[m[K [01;36m[Knote: [m[K...this statement, but the latter is misleadingly indented as if it were guarded by the ‘[01m[Kif[m[K’
             if (l.x_gpu)                   cuda_free(l.x_gpu);  [01;36m[Kl[m[K.x_gpu = NULL;
                                                                 [01;36m[K^[m[K
    [ 34%] [32mBuilding C object CMakeFiles/dark.dir/src/crop_layer.c.o[0m
    [ 35%] [32mBuilding C object CMakeFiles/dark.dir/src/dark_cuda.c.o[0m
    [ 35%] [32mBuilding C object CMakeFiles/dark.dir/src/data.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:[m[K In function ‘[01m[Kcudnn_check_error_extended[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:224:20:[m[K [01;35m[Kwarning: [m[Kcomparison between ‘[01m[KcudaError_t {aka enum cudaError}[m[K’ and ‘[01m[Kenum <anonymous>[m[K’ [[01;35m[K-Wenum-compare[m[K]
             if (status [01;35m[K!=[m[K CUDNN_STATUS_SUCCESS)
                        [01;35m[K^~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:[m[K In function ‘[01m[Kpre_allocate_pinned_memory[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:276:40:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%u[m[K’ expects argument of type ‘[01m[Kunsigned int[m[K’, but argument 2 has type ‘[01m[Klong unsigned int[m[K’ [[01;35m[K-Wformat=[m[K]
             printf("pre_allocate: size = [01;35m[K%Iu[m[K MB, num_of_blocks = %Iu, block_size = %Iu MB \n",
                                          [01;35m[K~~^[m[K
                                          [32m[K%Ilu[m[K
                 [32m[Ksize / (1024*1024)[m[K, num_of_blocks, pinned_block_size / (1024 * 1024));
                 [32m[K~~~~~~~~~~~~~~~~~~[m[K          
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:276:64:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%u[m[K’ expects argument of type ‘[01m[Kunsigned int[m[K’, but argument 3 has type ‘[01m[Ksize_t {aka const long unsigned int}[m[K’ [[01;35m[K-Wformat=[m[K]
             printf("pre_allocate: size = %Iu MB, num_of_blocks = [01;35m[K%Iu[m[K, block_size = %Iu MB \n",
                                                                  [01;35m[K~~^[m[K
                                                                  [32m[K%Ilu[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:276:82:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%u[m[K’ expects argument of type ‘[01m[Kunsigned int[m[K’, but argument 4 has type ‘[01m[Klong unsigned int[m[K’ [[01;35m[K-Wformat=[m[K]
             printf("pre_allocate: size = %Iu MB, num_of_blocks = %Iu, block_size = [01;35m[K%Iu[m[K MB \n",
                                                                                    [01;35m[K~~^[m[K
                                                                                    [32m[K%Ilu[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:286:37:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%d[m[K’ expects argument of type ‘[01m[Kint[m[K’, but argument 2 has type ‘[01m[Ksize_t {aka const long unsigned int}[m[K’ [[01;35m[K-Wformat=[m[K]
                     printf(" Allocated [01;35m[K%d[m[K pinned block \n", pinned_block_size);
                                        [01;35m[K~^[m[K
                                        [32m[K%ld[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:[m[K In function ‘[01m[Kcuda_make_array_pinned_preallocated[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:307:43:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%d[m[K’ expects argument of type ‘[01m[Kint[m[K’, but argument 2 has type ‘[01m[Ksize_t {aka long unsigned int}[m[K’ [[01;35m[K-Wformat=[m[K]
                 printf("\n Pinned block_id = [01;35m[K%d[m[K, filled = %f %% \n", pinned_block_id, filled);
                                              [01;35m[K~^[m[K
                                              [32m[K%ld[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:322:64:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%d[m[K’ expects argument of type ‘[01m[Kint[m[K’, but argument 2 has type ‘[01m[Klong unsigned int[m[K’ [[01;35m[K-Wformat=[m[K]
                 printf("Try to allocate new pinned memory, size = [01;35m[K%d[m[K MB \n", [32m[Ksize / (1024 * 1024)[m[K);
                                                                   [01;35m[K~^[m[K         [32m[K~~~~~~~~~~~~~~~~~~~~[m[K
                                                                   [32m[K%ld[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dark_cuda.c:328:63:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%d[m[K’ expects argument of type ‘[01m[Kint[m[K’, but argument 2 has type ‘[01m[Klong unsigned int[m[K’ [[01;35m[K-Wformat=[m[K]
                 printf("Try to allocate new pinned BLOCK, size = [01;35m[K%d[m[K MB \n", [32m[Ksize / (1024 * 1024)[m[K);
                                                                  [01;35m[K~^[m[K         [32m[K~~~~~~~~~~~~~~~~~~~~[m[K
                                                                  [32m[K%ld[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/data.c:[m[K In function ‘[01m[Kload_data_detection[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/data.c:1238:24:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kx[m[K’ [[01;35m[K-Wunused-variable[m[K]
                     int k, [01;35m[Kx[m[K, y;
                            [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/data.c:1038:43:[m[K [01;35m[Kwarning: [m[Kvariable ‘[01m[Kr_scale[m[K’ set but not used [[01;35m[K-Wunused-but-set-variable[m[K]
         float r1 = 0, r2 = 0, r3 = 0, r4 = 0, [01;35m[Kr_scale[m[K = 0;
                                               [01;35m[K^~~~~~~[m[K
    [ 35%] [32mBuilding C object CMakeFiles/darknet.dir/src/list.c.o[0m
    [ 36%] [32mBuilding C object CMakeFiles/dark.dir/src/deconvolutional_layer.c.o[0m
    [ 37%] [32mBuilding C object CMakeFiles/dark.dir/src/demo.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/demo.c:[m[K In function ‘[01m[Kdetect_in_thread[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/demo.c:102:16:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kprediction[m[K’ [[01;35m[K-Wunused-variable[m[K]
             float *[01;35m[Kprediction[m[K = network_predict(net, X);
                    [01;35m[K^~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/demo.c:100:15:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kl[m[K’ [[01;35m[K-Wunused-variable[m[K]
             layer [01;35m[Kl[m[K = net.layers[net.n - 1];
                   [01;35m[K^[m[K
    [ 37%] [32mBuilding C object CMakeFiles/dark.dir/src/detection_layer.c.o[0m
    [ 38%] [32mBuilding C object CMakeFiles/dark.dir/src/detector.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:[m[K In function ‘[01m[Kprint_cocos[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:453:29:[m[K [01;35m[Kwarning: [m[Kformat not a string literal and no format arguments [[01;35m[K-Wformat-security[m[K]
                     fprintf(fp, [01;35m[Kbuff[m[K);
                                 [01;35m[K^~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:[m[K In function ‘[01m[Keliminate_bdd[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:546:21:[m[K [01;35m[Kwarning: [m[Kstatement with no effect [[01;35m[K-Wunused-value[m[K]
                         [01;35m[Kfor[m[K (k; buf[k + n] != '\0'; k++)
                         [01;35m[K^~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:[m[K In function ‘[01m[Kvalidate_detector[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:659:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kmkd2[m[K’ [[01;35m[K-Wunused-variable[m[K]
             int [01;35m[Kmkd2[m[K = make_directory(buff2, 0777);
                 [01;35m[K^~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:657:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kmkd[m[K’ [[01;35m[K-Wunused-variable[m[K]
             int [01;35m[Kmkd[m[K = make_directory(buff, 0777);
                 [01;35m[K^~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:[m[K In function ‘[01m[Kvalidate_detector_map[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:1278:15:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kclass_recall[m[K’ [[01;35m[K-Wunused-variable[m[K]
             float [01;35m[Kclass_recall[m[K = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)(truth_classes_count[i] - tp_for_thresh_per_class[i]));
                   [01;35m[K^~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:1277:15:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kclass_precision[m[K’ [[01;35m[K-Wunused-variable[m[K]
             float [01;35m[Kclass_precision[m[K = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)fp_for_thresh_per_class[i]);
                   [01;35m[K^~~~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:[m[K In function ‘[01m[Kdraw_object[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/detector.c:1794:19:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kinv_loss[m[K’ [[01;35m[K-Wunused-variable[m[K]
                 float [01;35m[Kinv_loss[m[K = 1.0 / max_val_cmp(0.01, avg_loss);
                       [01;35m[K^~~~~~~~[m[K
    [ 39%] [32mBuilding C object CMakeFiles/dark.dir/src/dice.c.o[0m
    [ 39%] [32mBuilding C object CMakeFiles/dark.dir/src/dropout_layer.c.o[0m
    [ 40%] [32mBuilding C object CMakeFiles/darknet.dir/src/local_layer.c.o[0m
    [ 41%] [32mBuilding C object CMakeFiles/dark.dir/src/gaussian_yolo_layer.c.o[0m
    [ 41%] [32mBuilding C object CMakeFiles/dark.dir/src/gemm.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:[m[K In function ‘[01m[Kmake_gaussian_yolo_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:71:38:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
         if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Kl.output, batch*l.outputs * sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
                                          [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:7[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:78:38:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
         if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Kl.delta, batch*l.outputs * sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
                                          [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:7[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:[m[K In function ‘[01m[Kresize_gaussian_yolo_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:110:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess != cudaHostAlloc([01;35m[K&[m[Kl->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:7[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:119:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess != cudaHostAlloc([01;35m[K&[m[Kl->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gaussian_yolo_layer.c:7[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [ 42%] [32mBuilding C object CMakeFiles/dark.dir/src/go.c.o[0m
    [ 42%] [32mBuilding C object CMakeFiles/darknet.dir/src/lstm_layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:[m[K In function ‘[01m[K_castu32_f32[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:550:5:[m[K [01;35m[Kwarning: [m[Kdereferencing type-punned pointer will break strict-aliasing rules [[01;35m[K-Wstrict-aliasing[m[K]
         [01;35m[Kreturn[m[K *((float *)&a);
         [01;35m[K^~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:[m[K In function ‘[01m[Kconvolution_2d[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1066:27:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Ksum[m[K’ [[01;35m[K-Wunused-variable[m[K]
                         float [01;35m[Ksum[m[K = 0;
                               [01;35m[K^~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1046:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kall256i_one[m[K’ [[01;35m[K-Wunused-variable[m[K]
         __m256i [01;35m[Kall256i_one[m[K = _mm256_set1_epi32(1);
                 [01;35m[K^~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1045:12:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kall256_one[m[K’ [[01;35m[K-Wunused-variable[m[K]
         __m256 [01;35m[Kall256_one[m[K = _mm256_set1_ps(1);
                [01;35m[K^~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1043:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kidx256[m[K’ [[01;35m[K-Wunused-variable[m[K]
         __m256i [01;35m[Kidx256[m[K = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
                 [01;35m[K^~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1040:13:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kall256_last_zero[m[K’ [[01;35m[K-Wunused-variable[m[K]
         __m256i [01;35m[Kall256_last_zero[m[K =
                 [01;35m[K^~~~~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:[m[K In function ‘[01m[Kim2col_cpu_custom_bin[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/gemm.c:1644:17:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kall256_sing1[m[K’ [[01;35m[K-Wunused-variable[m[K]
             __m256i [01;35m[Kall256_sing1[m[K = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
                     [01;35m[K^~~~~~~~~~~~[m[K
    [ 42%] [32mBuilding C object CMakeFiles/dark.dir/src/gru_layer.c.o[0m
    [ 43%] [32mBuilding C object CMakeFiles/dark.dir/src/im2col.c.o[0m
    [ 44%] [32mBuilding C object CMakeFiles/darknet.dir/src/matrix.c.o[0m
    [ 45%] [32mBuilding C object CMakeFiles/dark.dir/src/image.c.o[0m
    [ 45%] [32mBuilding C object CMakeFiles/dark.dir/src/layer.c.o[0m
    [ 46%] [32mBuilding C object CMakeFiles/dark.dir/src/list.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.c:[m[K In function ‘[01m[Kfree_layer_custom[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.c:191:9:[m[K [01;35m[Kwarning: [m[Kthis ‘[01m[Kif[m[K’ clause does not guard... [[01;35m[K-Wmisleading-indentation[m[K]
             [01;35m[Kif[m[K (l.x_gpu)                   cuda_free(l.x_gpu);  l.x_gpu = NULL;
             [01;35m[K^~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.c:191:61:[m[K [01;36m[Knote: [m[K...this statement, but the latter is misleadingly indented as if it were guarded by the ‘[01m[Kif[m[K’
             if (l.x_gpu)                   cuda_free(l.x_gpu);  [01;36m[Kl[m[K.x_gpu = NULL;
                                                                 [01;36m[K^[m[K
    [ 47%] [32mBuilding C object CMakeFiles/darknet.dir/src/maxpool_layer.c.o[0m
    [ 47%] [32mBuilding C object CMakeFiles/dark.dir/src/local_layer.c.o[0m
    [ 48%] [32mBuilding C object CMakeFiles/dark.dir/src/lstm_layer.c.o[0m
    [ 49%] [32mBuilding C object CMakeFiles/dark.dir/src/matrix.c.o[0m
    [ 49%] [32mBuilding C object CMakeFiles/darknet.dir/src/network.c.o[0m
    [ 49%] [32mBuilding C object CMakeFiles/dark.dir/src/maxpool_layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network.c:[m[K In function ‘[01m[Kresize_network[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network.c:610:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Knet->input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped))
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network.c:1[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [ 50%] [32mBuilding C object CMakeFiles/darknet.dir/src/nightmare.c.o[0m
    [ 51%] [32mBuilding C object CMakeFiles/dark.dir/src/network.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network.c:[m[K In function ‘[01m[Kresize_network[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network.c:610:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Knet->input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped))
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network.c:1[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [ 51%] [32mBuilding C object CMakeFiles/dark.dir/src/nightmare.c.o[0m
    [ 51%] [32mBuilding C object CMakeFiles/darknet.dir/src/normalization_layer.c.o[0m
    [ 52%] [32mBuilding C object CMakeFiles/dark.dir/src/normalization_layer.c.o[0m
    [ 53%] [32mBuilding C object CMakeFiles/dark.dir/src/option_list.c.o[0m
    [ 54%] [32mBuilding C object CMakeFiles/darknet.dir/src/option_list.c.o[0m
    [ 55%] [32mBuilding C object CMakeFiles/darknet.dir/src/parser.c.o[0m
    [ 55%] [32mBuilding C object CMakeFiles/dark.dir/src/parser.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/parser.c:[m[K In function ‘[01m[Kparse_network_cfg_custom[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/parser.c:1581:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Knet.input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped)) net.input_pinned_cpu_flag = 1;
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.h:3[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activation_layer.h:4[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/parser.c:6[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [ 55%] [32mBuilding C object CMakeFiles/darknet.dir/src/region_layer.c.o[0m
    [ 56%] [32mBuilding C object CMakeFiles/darknet.dir/src/reorg_layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/parser.c:[m[K In function ‘[01m[Kparse_network_cfg_custom[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/parser.c:1581:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Knet.input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped)) net.input_pinned_cpu_flag = 1;
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.h:3[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activation_layer.h:4[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/parser.c:6[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/region_layer.c:[m[K In function ‘[01m[Kresize_region_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/region_layer.c:58:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kold_h[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kold_h[m[K = l->h;
             [01;35m[K^~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/region_layer.c:57:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kold_w[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kold_w[m[K = l->w;
             [01;35m[K^~~~~[m[K
    [ 56%] [32mBuilding C object CMakeFiles/darknet.dir/src/reorg_old_layer.c.o[0m
    [ 57%] [32mBuilding C object CMakeFiles/darknet.dir/src/rnn.c.o[0m
    [ 58%] [32mBuilding C object CMakeFiles/dark.dir/src/region_layer.c.o[0m
    [ 59%] [32mBuilding C object CMakeFiles/darknet.dir/src/rnn_layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/region_layer.c:[m[K In function ‘[01m[Kresize_region_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/region_layer.c:58:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kold_h[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kold_h[m[K = l->h;
             [01;35m[K^~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/region_layer.c:57:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kold_w[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kold_w[m[K = l->w;
             [01;35m[K^~~~~[m[K
    [ 59%] [32mBuilding C object CMakeFiles/darknet.dir/src/rnn_vid.c.o[0m
    [ 60%] [32mBuilding C object CMakeFiles/darknet.dir/src/route_layer.c.o[0m
    [ 60%] [32mBuilding C object CMakeFiles/darknet.dir/src/sam_layer.c.o[0m
    [ 61%] [32mBuilding C object CMakeFiles/darknet.dir/src/scale_channels_layer.c.o[0m
    [ 62%] [32mBuilding C object CMakeFiles/darknet.dir/src/shortcut_layer.c.o[0m
    [ 62%] [32mBuilding C object CMakeFiles/darknet.dir/src/softmax_layer.c.o[0m
    [ 62%] [32mBuilding C object CMakeFiles/dark.dir/src/reorg_layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/shortcut_layer.c:[m[K In function ‘[01m[Kmake_shortcut_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/shortcut_layer.c:55:15:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kscale[m[K’ [[01;35m[K-Wunused-variable[m[K]
             float [01;35m[Kscale[m[K = sqrt(2. / l.nweights);
                   [01;35m[K^~~~~[m[K
    [ 63%] [32mBuilding C object CMakeFiles/darknet.dir/src/super.c.o[0m
    [ 63%] [32mBuilding C object CMakeFiles/darknet.dir/src/swag.c.o[0m
    [ 64%] [32mBuilding C object CMakeFiles/dark.dir/src/reorg_old_layer.c.o[0m
    [ 65%] [32mBuilding C object CMakeFiles/darknet.dir/src/tag.c.o[0m
    [ 66%] [32mBuilding C object CMakeFiles/darknet.dir/src/tree.c.o[0m
    [ 66%] [32mBuilding C object CMakeFiles/darknet.dir/src/upsample_layer.c.o[0m
    [ 67%] [32mBuilding C object CMakeFiles/dark.dir/src/rnn.c.o[0m
    [ 68%] [32mBuilding C object CMakeFiles/darknet.dir/src/utils.c.o[0m
    [ 68%] [32mBuilding C object CMakeFiles/darknet.dir/src/voxel.c.o[0m
    [ 69%] [32mBuilding C object CMakeFiles/darknet.dir/src/writing.c.o[0m
    [ 69%] [32mBuilding C object CMakeFiles/dark.dir/src/rnn_layer.c.o[0m
    [ 70%] [32mBuilding C object CMakeFiles/darknet.dir/src/yolo.c.o[0m
    [ 71%] [32mBuilding C object CMakeFiles/dark.dir/src/rnn_vid.c.o[0m
    [ 71%] [32mBuilding C object CMakeFiles/dark.dir/src/route_layer.c.o[0m
    [ 72%] [32mBuilding C object CMakeFiles/dark.dir/src/sam_layer.c.o[0m
    [ 73%] [32mBuilding C object CMakeFiles/dark.dir/src/scale_channels_layer.c.o[0m
    [ 73%] [32mBuilding C object CMakeFiles/dark.dir/src/shortcut_layer.c.o[0m
    [ 73%] [32mBuilding C object CMakeFiles/darknet.dir/src/yolo_layer.c.o[0m
    [ 74%] [32mBuilding C object CMakeFiles/dark.dir/src/softmax_layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/shortcut_layer.c:[m[K In function ‘[01m[Kmake_shortcut_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/shortcut_layer.c:55:15:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kscale[m[K’ [[01;35m[K-Wunused-variable[m[K]
             float [01;35m[Kscale[m[K = sqrt(2. / l.nweights);
                   [01;35m[K^~~~~[m[K
    [ 75%] [32mBuilding CXX object CMakeFiles/darknet.dir/src/http_stream.cpp.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:[m[K In function ‘[01m[Kmake_yolo_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:62:38:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
         if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Kl.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
                                          [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.h:3[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.h:4[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:1[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:69:38:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
         if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Kl.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
                                          [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.h:3[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.h:4[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:1[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:[m[K In function ‘[01m[Kresize_yolo_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:96:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess != cudaHostAlloc([01;35m[K&[m[Kl->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.h:3[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.h:4[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:1[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:105:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess != cudaHostAlloc([01;35m[K&[m[Kl->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.h:3[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.h:4[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:1[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [ 75%] [32mBuilding CXX object CMakeFiles/darknet.dir/src/image_opencv.cpp.o[0m
    [ 75%] [32mBuilding C object CMakeFiles/dark.dir/src/super.c.o[0m
    [ 76%] [32mBuilding C object CMakeFiles/dark.dir/src/swag.c.o[0m
    [ 77%] [32mBuilding CUDA object CMakeFiles/darknet.dir/src/activation_kernels.cu.o[0m
    [ 78%] [32mBuilding C object CMakeFiles/dark.dir/src/tag.c.o[0m
    [ 78%] [32mBuilding C object CMakeFiles/dark.dir/src/tree.c.o[0m
    In file included from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:580:0[m[K:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/httplib.h:129:0:[m[K [01;35m[Kwarning: [m[K"INVALID_SOCKET" redefined
     #define INVALID_SOCKET (-1)
     
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:73:0:[m[K [01;36m[Knote: [m[Kthis is the location of the previous definition
     #define INVALID_SOCKET -1
     
    [ 79%] [32mBuilding C object CMakeFiles/dark.dir/src/upsample_layer.c.o[0m
    [ 79%] [32mBuilding C object CMakeFiles/dark.dir/src/utils.c.o[0m
    [ 80%] [32mBuilding C object CMakeFiles/dark.dir/src/voxel.c.o[0m
    [ 81%] [32mBuilding CUDA object CMakeFiles/darknet.dir/src/avgpool_layer_kernels.cu.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:[m[K In member function ‘[01m[Kbool JSON_sender::write(const char*)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:249:21:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kn[m[K’ [[01;35m[K-Wunused-variable[m[K]
                     int [01;35m[Kn[m[K = _write(client, outputbuf, outlen);
                         [01;35m[K^[m[K
    [ 82%] [32mBuilding C object CMakeFiles/dark.dir/src/writing.c.o[0m
    [ 82%] [32mBuilding C object CMakeFiles/dark.dir/src/yolo.c.o[0m
    [ 83%] [32mBuilding C object CMakeFiles/dark.dir/src/yolo_layer.c.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:[m[K In function ‘[01m[Kmake_yolo_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:62:38:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
         if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Kl.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
                                          [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.h:3[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.h:4[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:1[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:69:38:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
         if (cudaSuccess == cudaHostAlloc([01;35m[K&[m[Kl.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
                                          [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.h:3[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.h:4[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:1[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:[m[K In function ‘[01m[Kresize_yolo_layer[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:96:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess != cudaHostAlloc([01;35m[K&[m[Kl->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.h:3[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.h:4[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:1[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:105:42:[m[K [01;35m[Kwarning: [m[Kpassing argument 1 of ‘[01m[KcudaHostAlloc[m[K’ from incompatible pointer type [[01;35m[K-Wincompatible-pointer-types[m[K]
             if (cudaSuccess != cudaHostAlloc([01;35m[K&[m[Kl->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
                                              [01;35m[K^[m[K
    In file included from [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h:96:0[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/darknet.h:41[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/activations.h:3[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/layer.h:4[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.h:5[m[K,
                     from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_layer.c:1[m[K:
    [01m[K/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h:4391:39:[m[K [01;36m[Knote: [m[Kexpected ‘[01m[Kvoid **[m[K’ but argument is of type ‘[01m[Kfloat **[m[K’
     extern __host__ cudaError_t CUDARTAPI [01;36m[KcudaHostAlloc[m[K(void **pHost, size_t size, unsigned int flags);
                                           [01;36m[K^~~~~~~~~~~~~[m[K
    [ 83%] [32mBuilding CUDA object CMakeFiles/darknet.dir/src/blas_kernels.cu.o[0m
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas_kernels.cu(1086): warning: variable "out_index" was declared but never referenced
    
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas_kernels.cu(1130): warning: variable "step" was set but never used
    
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas_kernels.cu(1734): warning: variable "stage_id" was declared but never referenced
    
    [ 84%] [32mBuilding CUDA object CMakeFiles/darknet.dir/src/col2im_kernels.cu.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:[m[K In function ‘[01m[Kvoid draw_detections_cv_v3(void**, detection*, int, float, char**, image**, int, int)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:910:23:[m[K [01;35m[Kwarning: [m[Kvariable ‘[01m[Krgb[m[K’ set but not used [[01;35m[K-Wunused-but-set-variable[m[K]
                     float [01;35m[Krgb[m[K[3];
                           [01;35m[K^~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:[m[K In function ‘[01m[Kvoid cv_draw_object(image, float*, int, int, int*, float*, int*, int, char**)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:1391:14:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kbuff[m[K’ [[01;35m[K-Wunused-variable[m[K]
             char [01;35m[Kbuff[m[K[100];
                  [01;35m[K^~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:1367:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kit_tb_res[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kit_tb_res[m[K = cv::createTrackbar(it_trackbar_name, window_name, &it_trackbar_value, 1000);
             [01;35m[K^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:1371:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Klr_tb_res[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Klr_tb_res[m[K = cv::createTrackbar(lr_trackbar_name, window_name, &lr_trackbar_value, 20);
             [01;35m[K^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:1375:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kcl_tb_res[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kcl_tb_res[m[K = cv::createTrackbar(cl_trackbar_name, window_name, &cl_trackbar_value, classes-1);
             [01;35m[K^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:1378:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kbo_tb_res[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kbo_tb_res[m[K = cv::createTrackbar(bo_trackbar_name, window_name, boxonly, 1);
             [01;35m[K^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:[m[K In member function ‘[01m[Kbool MJPG_sender::write(const cv::Mat&)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:507:113:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%zu[m[K’ expects argument of type ‘[01m[Ksize_t[m[K’, but argument 3 has type ‘[01m[Kint[m[K’ [[01;35m[K-Wformat=[m[K]
                     sprintf(head, "--mjpegstream\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", outlen[01;35m[K)[m[K;
                                                                                                                     [01;35m[K^[m[K
    [ 84%] [32mBuilding CXX object CMakeFiles/dark.dir/src/http_stream.cpp.o[0m
    [ 85%] [32mBuilding CXX object CMakeFiles/dark.dir/src/image_opencv.cpp.o[0m
    In file included from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:580:0[m[K:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/httplib.h:129:0:[m[K [01;35m[Kwarning: [m[K"INVALID_SOCKET" redefined
     #define INVALID_SOCKET (-1)
     
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:73:0:[m[K [01;36m[Knote: [m[Kthis is the location of the previous definition
     #define INVALID_SOCKET -1
     
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:[m[K In member function ‘[01m[Kbool JSON_sender::write(const char*)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:249:21:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kn[m[K’ [[01;35m[K-Wunused-variable[m[K]
                     int [01;35m[Kn[m[K = _write(client, outputbuf, outlen);
                         [01;35m[K^[m[K
    [ 85%] [32mBuilding CUDA object CMakeFiles/darknet.dir/src/convolutional_kernels.cu.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:[m[K In member function ‘[01m[Kbool MJPG_sender::write(const cv::Mat&)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/http_stream.cpp:507:113:[m[K [01;35m[Kwarning: [m[Kformat ‘[01m[K%zu[m[K’ expects argument of type ‘[01m[Ksize_t[m[K’, but argument 3 has type ‘[01m[Kint[m[K’ [[01;35m[K-Wformat=[m[K]
                     sprintf(head, "--mjpegstream\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", outlen[01;35m[K)[m[K;
                                                                                                                     [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:[m[K In function ‘[01m[Kvoid draw_detections_cv_v3(void**, detection*, int, float, char**, image**, int, int)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:910:23:[m[K [01;35m[Kwarning: [m[Kvariable ‘[01m[Krgb[m[K’ set but not used [[01;35m[K-Wunused-but-set-variable[m[K]
                     float [01;35m[Krgb[m[K[3];
                           [01;35m[K^~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:[m[K In function ‘[01m[Kvoid cv_draw_object(image, float*, int, int, int*, float*, int*, int, char**)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:1391:14:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kbuff[m[K’ [[01;35m[K-Wunused-variable[m[K]
             char [01;35m[Kbuff[m[K[100];
                  [01;35m[K^~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:1367:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kit_tb_res[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kit_tb_res[m[K = cv::createTrackbar(it_trackbar_name, window_name, &it_trackbar_value, 1000);
             [01;35m[K^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:1371:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Klr_tb_res[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Klr_tb_res[m[K = cv::createTrackbar(lr_trackbar_name, window_name, &lr_trackbar_value, 20);
             [01;35m[K^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:1375:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kcl_tb_res[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kcl_tb_res[m[K = cv::createTrackbar(cl_trackbar_name, window_name, &cl_trackbar_value, classes-1);
             [01;35m[K^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/image_opencv.cpp:1378:9:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kbo_tb_res[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int [01;35m[Kbo_tb_res[m[K = cv::createTrackbar(bo_trackbar_name, window_name, boxonly, 1);
             [01;35m[K^~~~~~~~~[m[K
    [ 86%] [32mBuilding CUDA object CMakeFiles/darknet.dir/src/crop_layer_kernels.cu.o[0m
    [ 87%] [32mBuilding CUDA object CMakeFiles/darknet.dir/src/deconvolutional_kernels.cu.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas_kernels.cu:[m[K In function ‘[01m[Kvoid backward_shortcut_multilayer_gpu(int, int, int, int*, float**, float*, float*, float*, float*, int, float*, float**, WEIGHTS_NORMALIZATION_T)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas_kernels.cu:1130:5:[m[K [01;35m[Kwarning: [m[Kvariable ‘[01m[Kstep[m[K’ set but not used [[01;35m[K-Wunused-but-set-variable[m[K]
         [01;35m[Kint [m[Kstep = 0;
         [01;35m[K^~~~[m[K
    [ 88%] [32mBuilding CUDA object CMakeFiles/dark.dir/src/activation_kernels.cu.o[0m
    [ 88%] [32mBuilding CUDA object CMakeFiles/darknet.dir/src/dropout_layer_kernels.cu.o[0m
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dropout_layer_kernels.cu(140): warning: variable "cur_scale" was declared but never referenced
    
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dropout_layer_kernels.cu(245): warning: variable "cur_scale" was declared but never referenced
    
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dropout_layer_kernels.cu(262): warning: variable "block_prob" was declared but never referenced
    
    [ 89%] [32mBuilding CUDA object CMakeFiles/darknet.dir/src/im2col_kernels.cu.o[0m
    [ 89%] [32mBuilding CUDA object CMakeFiles/darknet.dir/src/maxpool_layer_kernels.cu.o[0m
    [ 90%] [32mBuilding CUDA object CMakeFiles/darknet.dir/src/network_kernels.cu.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/im2col_kernels.cu:125:18:[m[K [01;35m[Kwarning: [m[K"/*" within comment [[01;35m[K-Wcomment[m[K]
                     //*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                       
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/im2col_kernels.cu:1178:6:[m[K [01;35m[Kwarning: [m[K"/*" within comment [[01;35m[K-Wcomment[m[K]
         //*((uint64_t *)(A_s + (local_i*lda + k) / 8)) = *((uint64_t *)(A + (i_cur*lda + k) / 8));    // weights
           
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network_kernels.cu(359): warning: variable "l" was declared but never referenced
    
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network_kernels.cu(368): warning: variable "im" was set but never used
    
    [ 90%] [32mBuilding CUDA object CMakeFiles/dark.dir/src/avgpool_layer_kernels.cu.o[0m
    [ 91%] [32mBuilding CUDA object CMakeFiles/dark.dir/src/blas_kernels.cu.o[0m
    [ 91%] [32mBuilding CUDA object CMakeFiles/dark.dir/src/col2im_kernels.cu.o[0m
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas_kernels.cu(1086): warning: variable "out_index" was declared but never referenced
    
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas_kernels.cu(1130): warning: variable "step" was set but never used
    
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network_kernels.cu:[m[K In function ‘[01m[Kfloat train_network_datum_gpu(network, float*, float*)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network_kernels.cu:359:7:[m[K [01;35m[Kwarning: [m[Kvariable ‘[01m[Kl[m[K’ set but not used [[01;35m[K-Wunused-but-set-variable[m[K]
           [01;35m[K [m[K layer l = net.layers[net.n - 1];
           [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network_kernels.cu:368:7:[m[K [01;35m[Kwarning: [m[Kvariable ‘[01m[Kim[m[K’ set but not used [[01;35m[K-Wunused-but-set-variable[m[K]
           [01;35m[K  [m[Kimage im;
           [01;35m[K^[m[K 
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas_kernels.cu(1734): warning: variable "stage_id" was declared but never referenced
    
    [ 92%] [32mBuilding CUDA object CMakeFiles/dark.dir/src/convolutional_kernels.cu.o[0m
    [ 93%] [32mBuilding CUDA object CMakeFiles/dark.dir/src/crop_layer_kernels.cu.o[0m
    [ 93%] [32mBuilding CUDA object CMakeFiles/dark.dir/src/deconvolutional_kernels.cu.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/im2col_kernels.cu:125:18:[m[K [01;35m[Kwarning: [m[K"/*" within comment [[01;35m[K-Wcomment[m[K]
                     //*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                       
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/im2col_kernels.cu:1178:6:[m[K [01;35m[Kwarning: [m[K"/*" within comment [[01;35m[K-Wcomment[m[K]
         //*((uint64_t *)(A_s + (local_i*lda + k) / 8)) = *((uint64_t *)(A + (i_cur*lda + k) / 8));    // weights
           
    [ 94%] [32mBuilding CUDA object CMakeFiles/dark.dir/src/dropout_layer_kernels.cu.o[0m
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dropout_layer_kernels.cu(140): warning: variable "cur_scale" was declared but never referenced
    
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dropout_layer_kernels.cu(245): warning: variable "cur_scale" was declared but never referenced
    
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/dropout_layer_kernels.cu(262): warning: variable "block_prob" was declared but never referenced
    
    [ 94%] [32mBuilding CUDA object CMakeFiles/dark.dir/src/im2col_kernels.cu.o[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/im2col_kernels.cu:125:18:[m[K [01;35m[Kwarning: [m[K"/*" within comment [[01;35m[K-Wcomment[m[K]
                     //*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                       
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/im2col_kernels.cu:1178:6:[m[K [01;35m[Kwarning: [m[K"/*" within comment [[01;35m[K-Wcomment[m[K]
         //*((uint64_t *)(A_s + (local_i*lda + k) / 8)) = *((uint64_t *)(A + (i_cur*lda + k) / 8));    // weights
           
    [ 95%] [32mBuilding CUDA object CMakeFiles/dark.dir/src/maxpool_layer_kernels.cu.o[0m
    [ 96%] [32mBuilding CUDA object CMakeFiles/dark.dir/src/network_kernels.cu.o[0m
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network_kernels.cu(359): warning: variable "l" was declared but never referenced
    
    /content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network_kernels.cu(368): warning: variable "im" was set but never used
    
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas_kernels.cu:[m[K In function ‘[01m[Kvoid backward_shortcut_multilayer_gpu(int, int, int, int*, float**, float*, float*, float*, float*, int, float*, float**, WEIGHTS_NORMALIZATION_T)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/blas_kernels.cu:1130:5:[m[K [01;35m[Kwarning: [m[Kvariable ‘[01m[Kstep[m[K’ set but not used [[01;35m[K-Wunused-but-set-variable[m[K]
         [01;35m[Kint [m[Kstep = 0;
         [01;35m[K^~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network_kernels.cu:[m[K In function ‘[01m[Kfloat train_network_datum_gpu(network, float*, float*)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network_kernels.cu:359:7:[m[K [01;35m[Kwarning: [m[Kvariable ‘[01m[Kl[m[K’ set but not used [[01;35m[K-Wunused-but-set-variable[m[K]
           [01;35m[K [m[K layer l = net.layers[net.n - 1];
           [01;35m[K^[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/network_kernels.cu:368:7:[m[K [01;35m[Kwarning: [m[Kvariable ‘[01m[Kim[m[K’ set but not used [[01;35m[K-Wunused-but-set-variable[m[K]
           [01;35m[K  [m[Kimage im;
           [01;35m[K^[m[K 
    [ 97%] [32m[1mLinking CUDA device code CMakeFiles/darknet.dir/cmake_device_link.o[0m
    [ 97%] [32m[1mLinking CXX executable darknet[0m
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/im2col_kernels.cu:125:18:[m[K [01;35m[Kwarning: [m[K"/*" within comment [[01;35m[K-Wcomment[m[K]
                     //*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                       
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/im2col_kernels.cu:1178:6:[m[K [01;35m[Kwarning: [m[K"/*" within comment [[01;35m[K-Wcomment[m[K]
         //*((uint64_t *)(A_s + (local_i*lda + k) / 8)) = *((uint64_t *)(A + (i_cur*lda + k) / 8));    // weights
           
    [ 97%] Built target darknet
    [ 97%] [32m[1mLinking CUDA device code CMakeFiles/dark.dir/cmake_device_link.o[0m
    [ 98%] [32m[1mLinking CXX shared library libdark.so[0m
    [ 98%] Built target dark
    [35m[1mScanning dependencies of target uselib[0m
    [ 99%] [32mBuilding CXX object CMakeFiles/uselib.dir/src/yolo_console_dll.cpp.o[0m
    In file included from [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_console_dll.cpp:23:0[m[K:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:[m[K In constructor ‘[01m[Ktrack_kalman_t::track_kalman_t(int, int, float, cv::Size)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:708:14:[m[K [01;35m[Kwarning: [m[K‘[01m[Ktrack_kalman_t::img_size[m[K’ will be initialized after [[01;35m[K-Wreorder[m[K]
         cv::Size [01;35m[Kimg_size[m[K;  // max value of x,y,w,h
                  [01;35m[K^~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:700:9:[m[K [01;35m[Kwarning: [m[K  ‘[01m[Kint track_kalman_t::track_id_counter[m[K’ [[01;35m[K-Wreorder[m[K]
         int [01;35m[Ktrack_id_counter[m[K;
             [01;35m[K^~~~~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:853:5:[m[K [01;35m[Kwarning: [m[K  when initialized here [[01;35m[K-Wreorder[m[K]
         [01;35m[Ktrack_kalman_t[m[K(int _max_objects = 1000, int _min_frames = 3, float _max_dist = 40, cv::Size _img_size = cv::Size(10000, 10000)) :
         [01;35m[K^~~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:[m[K In member function ‘[01m[Kvoid track_kalman_t::clear_old_states()[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:879:50:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
                     if ((result_vec_pred[state_id].x > img_size.width) ||
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:880:50:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
                         (result_vec_pred[state_id].y > img_size.height))
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:[m[K In member function ‘[01m[Ktrack_kalman_t::tst_t track_kalman_t::get_state_id(bbox_t, std::vector<bool>&)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:900:30:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
             for (size_t i = 0; [01;35m[Ki < max_objects[m[K; ++i)
                                [01;35m[K~~^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:[m[K In member function ‘[01m[Kstd::vector<bbox_t> track_kalman_t::predict()[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:990:30:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
             for (size_t i = 0; [01;35m[Ki < max_objects[m[K; ++i)
                                [01;35m[K~~^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:[m[K In member function ‘[01m[Kstd::vector<bbox_t> track_kalman_t::correct(std::vector<bbox_t>)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/include/yolo_v2_class.hpp:1025:30:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
             for (size_t i = 0; [01;35m[Ki < max_objects[m[K; ++i)
                                [01;35m[K~~^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_console_dll.cpp:[m[K In function ‘[01m[Kvoid draw_boxes(cv::Mat, std::vector<bbox_t>, std::vector<std::__cxx11::basic_string<char> >, int, int)[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_console_dll.cpp:192:46:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
                 int max_width = ([01;35m[Ktext_size.width > i.w + 2[m[K) ? text_size.width : (i.w + 2);
                                  [01;35m[K~~~~~~~~~~~~~~~~^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_console_dll.cpp:201:62:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [[01;35m[K-Wsign-compare[m[K]
                     int const max_width_3d = ([01;35m[Ktext_size_3d.width > i.w + 2[m[K) ? text_size_3d.width : (i.w + 2);
                                               [01;35m[K~~~~~~~~~~~~~~~~~~~^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_console_dll.cpp:183:15:[m[K [01;35m[Kwarning: [m[Kunused variable ‘[01m[Kcolors[m[K’ [[01;35m[K-Wunused-variable[m[K]
         int const [01;35m[Kcolors[m[K[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
                   [01;35m[K^~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_console_dll.cpp:[m[K In constructor ‘[01m[Kmain(int, char**)::detection_data_t::detection_data_t()[m[K’:
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_console_dll.cpp:398:26:[m[K [01;35m[Kwarning: [m[K‘[01m[Kmain(int, char**)::detection_data_t::exit_flag[m[K’ will be initialized after [[01;35m[K-Wreorder[m[K]
                         bool [01;35m[Kexit_flag[m[K;
                              [01;35m[K^~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_console_dll.cpp:396:26:[m[K [01;35m[Kwarning: [m[K  ‘[01m[Kbool main(int, char**)::detection_data_t::new_detection[m[K’ [[01;35m[K-Wreorder[m[K]
                         bool [01;35m[Knew_detection[m[K;
                              [01;35m[K^~~~~~~~~~~~~[m[K
    [01m[K/content/Train-YOLOv4-with-OpenImagesV4-Colab/src/yolo_console_dll.cpp:401:21:[m[K [01;35m[Kwarning: [m[K  when initialized here [[01;35m[K-Wreorder[m[K]
                         [01;35m[Kdetection_data_t[m[K() : exit_flag(false), new_detection(false) {}
                         [01;35m[K^~~~~~~~~~~~~~~~[m[K
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
    

Download weights of YoloV4 pretrain from google drive


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



## Customize model config file

Copy config file


```
!cp cfg/yolov4-custom.cfg yolov4-custom.txt
```

Edit yolov4-custom.txt file as follows: 

- Line 3: Set batch=64.

- Line 4: Set subdivisions=32, the batch will be divided by 16 to decrease GPU VRAM requirements.

- Change line max_batches to classes*2000 but not less than number of training images, and not less than 6000, f.e. max_batches=6000 if you train for 3 classes.

- Change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400.

- Change line classes=2 to your number of objects in each of 3: Line 970 1058 1146.

- Change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers. Line 1139 1051 963.

Save that after making the changes.

Train Model


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
      10 conv     64       1 x 1/ 1    304 x 304 x 128 ->  304 x 304 x  64 1.514 BF
      11 conv    128       3 x 3/ 2    304 x 304 x  64 ->  152 x 152 x 128 3.407 BF
      12 conv     64       1 x 1/ 1    152 x 152 x 128 ->  152 x 152 x  64 0.379 BF
      13 route  11 		                           ->  152 x 152 x 128 
      14 conv     64       1 x 1/ 1    152 x 152 x 128 ->  152 x 152 x  64 0.379 BF
      15 conv     64       1 x 1/ 1    152 x 152 x  64 ->  152 x 152 x  64 0.189 BF
      16 conv     64       3 x 3/ 1    152 x 152 x  64 ->  152 x 152 x  64 1.703 BF
      17 Shortcut Layer: 14,  wt = 0, wn = 0, outputs: 152 x 152 x  64 0.001 BF
      18 conv     64       1 x 1/ 1    152 x 152 x  64 ->  152 x 152 x  64 0.189 BF
      19 conv     64       3 x 3/ 1    152 x 152 x  64 ->  152 x 152 x  64 1.703 BF
      20 Shortcut Layer: 17,  wt = 0, wn = 0, outputs: 152 x 152 x  64 0.001 BF
      21 conv     64       1 x 1/ 1    152 x 152 x  64 ->  152 x 152 x  64 0.189 BF
      22 route  21 12 	                           ->  152 x 152 x 128 
      23 conv    128       1 x 1/ 1    152 x 152 x 128 ->  152 x 152 x 128 0.757 BF
      24 conv    256       3 x 3/ 2    152 x 152 x 128 ->   76 x  76 x 256 3.407 BF
      25 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
      26 route  24 		                           ->   76 x  76 x 256 
      27 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
      28 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      29 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      30 Shortcut Layer: 27,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      31 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      32 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      33 Shortcut Layer: 30,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      34 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      35 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      36 Shortcut Layer: 33,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      37 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      38 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      39 Shortcut Layer: 36,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      40 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      41 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      42 Shortcut Layer: 39,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      43 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      44 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      45 Shortcut Layer: 42,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      46 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      47 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      48 Shortcut Layer: 45,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      49 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      50 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      51 Shortcut Layer: 48,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      52 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      53 route  52 25 	                           ->   76 x  76 x 256 
      54 conv    256       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 256 0.757 BF
      55 conv    512       3 x 3/ 2     76 x  76 x 256 ->   38 x  38 x 512 3.407 BF
      56 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
      57 route  55 		                           ->   38 x  38 x 512 
      58 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
      59 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      60 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      61 Shortcut Layer: 58,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      62 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      63 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      64 Shortcut Layer: 61,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      65 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      66 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      67 Shortcut Layer: 64,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      68 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      69 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      70 Shortcut Layer: 67,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      71 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      72 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      73 Shortcut Layer: 70,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      74 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      75 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      76 Shortcut Layer: 73,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      77 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      78 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      79 Shortcut Layer: 76,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      80 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      81 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      82 Shortcut Layer: 79,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      83 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      84 route  83 56 	                           ->   38 x  38 x 512 
      85 conv    512       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 512 0.757 BF
      86 conv   1024       3 x 3/ 2     38 x  38 x 512 ->   19 x  19 x1024 3.407 BF
      87 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
      88 route  86 		                           ->   19 x  19 x1024 
      89 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
      90 conv    512       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.189 BF
      91 conv    512       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x 512 1.703 BF
      92 Shortcut Layer: 89,  wt = 0, wn = 0, outputs:  19 x  19 x 512 0.000 BF
      93 conv    512       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.189 BF
      94 conv    512       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x 512 1.703 BF
      95 Shortcut Layer: 92,  wt = 0, wn = 0, outputs:  19 x  19 x 512 0.000 BF
      96 conv    512       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.189 BF
      97 conv    512       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x 512 1.703 BF
      98 Shortcut Layer: 95,  wt = 0, wn = 0, outputs:  19 x  19 x 512 0.000 BF
      99 conv    512       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.189 BF
     100 conv    512       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x 512 1.703 BF
     101 Shortcut Layer: 98,  wt = 0, wn = 0, outputs:  19 x  19 x 512 0.000 BF
     102 conv    512       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.189 BF
     103 route  102 87 	                           ->   19 x  19 x1024 
     104 conv   1024       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x1024 0.757 BF
     105 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     106 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
     107 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     108 max                5x 5/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.005 BF
     109 route  107 		                           ->   19 x  19 x 512 
     110 max                9x 9/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.015 BF
     111 route  107 		                           ->   19 x  19 x 512 
     112 max               13x13/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.031 BF
     113 route  112 110 108 107 	                   ->   19 x  19 x2048 
     114 conv    512       1 x 1/ 1     19 x  19 x2048 ->   19 x  19 x 512 0.757 BF
     115 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
     116 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     117 conv    256       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 256 0.095 BF
     118 upsample                 2x    19 x  19 x 256 ->   38 x  38 x 256
     119 route  85 		                           ->   38 x  38 x 512 
     120 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     121 route  120 118 	                           ->   38 x  38 x 512 
     122 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     123 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
     124 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     125 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
     126 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     127 conv    128       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 128 0.095 BF
     128 upsample                 2x    38 x  38 x 128 ->   76 x  76 x 128
     129 route  54 		                           ->   76 x  76 x 256 
     130 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
     131 route  130 128 	                           ->   76 x  76 x 256 
     132 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
     133 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
     134 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
     135 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
     136 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
     137 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
     138 conv     21       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x  21 0.062 BF
     139 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, cls_norm: 1.00, scale_x_y: 1.20
     140 route  136 		                           ->   76 x  76 x 128 
     141 conv    256       3 x 3/ 2     76 x  76 x 128 ->   38 x  38 x 256 0.852 BF
     142 route  141 126 	                           ->   38 x  38 x 512 
     143 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     144 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
     145 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     146 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
     147 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     148 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
     149 conv     21       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x  21 0.031 BF
     150 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, cls_norm: 1.00, scale_x_y: 1.10
     151 route  147 		                           ->   38 x  38 x 256 
     152 conv    512       3 x 3/ 2     38 x  38 x 256 ->   19 x  19 x 512 0.852 BF
     153 route  152 116 	                           ->   19 x  19 x1024 
     154 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     155 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
     156 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     157 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
     158 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     159 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
     160 conv     21       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x  21 0.016 BF
     161 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, cls_norm: 1.00, scale_x_y: 1.05
    Total BFLOPS 127.248 
    avg_outputs = 1046494 
     Allocate additional workspace_size = 118.88 MB 
    Loading weights from yolov4.conv.137...Done! Loaded 137 layers from weights-file 
     Create 6 permanent cpu-threads 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.495013, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9760.333984, iou_loss = 0.000000, total_loss = 9760.333984 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.354280, GIOU: 0.315922), Class: 0.518870, Obj: 0.519884, No Obj: 0.505231, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 2516.348145, iou_loss = 1.348877, total_loss = 2517.697021 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.376541, GIOU: 0.333236), Class: 0.474669, Obj: 0.493170, No Obj: 0.495870, .5R: 0.333333, .75R: 0.000000, count: 3, class_loss = 611.249634, iou_loss = 0.123779, total_loss = 611.373413 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.205704, GIOU: -0.061598), Class: 0.603062, Obj: 0.476906, No Obj: 0.494213, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 9709.390625, iou_loss = 5.637695, total_loss = 9715.028320 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.272507, GIOU: 0.272506), Class: 0.559083, Obj: 0.527297, No Obj: 0.506033, .5R: 0.000000, .75R: 0.000000, count: 2, class_loss = 2524.268311, iou_loss = 0.501465, total_loss = 2524.769775 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.275033, GIOU: 0.199859), Class: 0.503691, Obj: 0.465126, No Obj: 0.495925, .5R: 0.000000, .75R: 0.000000, count: 6, class_loss = 614.199219, iou_loss = 0.249207, total_loss = 614.448425 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.107726, GIOU: -0.235034), Class: 0.704929, Obj: 0.490540, No Obj: 0.494390, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9705.969727, iou_loss = 0.102539, total_loss = 9706.072266 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.404748, GIOU: 0.317747), Class: 0.478510, Obj: 0.499874, No Obj: 0.504526, .5R: 0.272727, .75R: 0.090909, count: 11, class_loss = 2512.256836, iou_loss = 10.812988, total_loss = 2523.069824 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.291954, GIOU: 0.124002), Class: 0.487209, Obj: 0.512588, No Obj: 0.494291, .5R: 0.071429, .75R: 0.000000, count: 14, class_loss = 612.235291, iou_loss = 0.417053, total_loss = 612.652344 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494468, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9718.691406, iou_loss = 0.000000, total_loss = 9718.691406 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.422022, GIOU: 0.305000), Class: 0.460699, Obj: 0.529639, No Obj: 0.505910, .5R: 0.375000, .75R: 0.000000, count: 8, class_loss = 2525.031738, iou_loss = 3.601807, total_loss = 2528.633545 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.358846, GIOU: 0.306865), Class: 0.521013, Obj: 0.527683, No Obj: 0.496872, .5R: 0.181818, .75R: 0.000000, count: 11, class_loss = 616.225891, iou_loss = 0.967529, total_loss = 617.193420 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.496449, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9826.120117, iou_loss = 0.000000, total_loss = 9826.120117 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.305670, GIOU: 0.082297), Class: 0.370553, Obj: 0.345646, No Obj: 0.505819, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2528.297607, iou_loss = 0.106934, total_loss = 2528.404541 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.423847, GIOU: 0.319732), Class: 0.547922, Obj: 0.478647, No Obj: 0.494263, .5R: 0.400000, .75R: 0.000000, count: 5, class_loss = 610.485352, iou_loss = 0.131531, total_loss = 610.616882 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.295472, GIOU: 0.136790), Class: 0.544923, Obj: 0.519539, No Obj: 0.495832, .5R: 0.148148, .75R: 0.000000, count: 54, class_loss = 9783.389648, iou_loss = 102.248047, total_loss = 9885.637695 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.301398, GIOU: 0.181325), Class: 0.566420, Obj: 0.503912, No Obj: 0.504640, .5R: 0.095238, .75R: 0.000000, count: 42, class_loss = 2522.553467, iou_loss = 33.129639, total_loss = 2555.683105 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.496175, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 612.963379, iou_loss = 0.000000, total_loss = 612.963379 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.308232, GIOU: 0.255640), Class: 0.514439, Obj: 0.548895, No Obj: 0.494966, .5R: 0.200000, .75R: 0.000000, count: 10, class_loss = 9737.953125, iou_loss = 20.726562, total_loss = 9758.679688 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.340247, GIOU: 0.143047), Class: 0.501677, Obj: 0.489039, No Obj: 0.504808, .5R: 0.142857, .75R: 0.000000, count: 21, class_loss = 2519.001465, iou_loss = 7.678711, total_loss = 2526.680176 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.358201, GIOU: 0.330286), Class: 0.497102, Obj: 0.518768, No Obj: 0.495042, .5R: 0.222222, .75R: 0.000000, count: 9, class_loss = 613.640991, iou_loss = 0.997498, total_loss = 614.638489 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494931, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9760.267578, iou_loss = 0.000000, total_loss = 9760.267578 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.162606, GIOU: 0.162606), Class: 0.427406, Obj: 0.498034, No Obj: 0.505616, .5R: 0.000000, .75R: 0.000000, count: 2, class_loss = 2521.010254, iou_loss = 0.106689, total_loss = 2521.116943 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.378593, GIOU: 0.330604), Class: 0.493254, Obj: 0.428424, No Obj: 0.496034, .5R: 0.285714, .75R: 0.142857, count: 7, class_loss = 614.719543, iou_loss = 0.418091, total_loss = 615.137634 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.496025, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9789.716797, iou_loss = 0.000000, total_loss = 9789.716797 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.371649, GIOU: 0.355883), Class: 0.494343, Obj: 0.479314, No Obj: 0.504982, .5R: 0.166667, .75R: 0.000000, count: 6, class_loss = 2514.052734, iou_loss = 3.159668, total_loss = 2517.212402 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.403665, GIOU: 0.401622), Class: 0.497558, Obj: 0.541186, No Obj: 0.494439, .5R: 0.400000, .75R: 0.200000, count: 5, class_loss = 609.242981, iou_loss = 1.313416, total_loss = 610.556396 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.225919, GIOU: 0.208631), Class: 0.570833, Obj: 0.582174, No Obj: 0.495052, .5R: 0.000000, .75R: 0.000000, count: 5, class_loss = 9761.231445, iou_loss = 7.199219, total_loss = 9768.430664 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.303535, GIOU: 0.121722), Class: 0.520424, Obj: 0.515454, No Obj: 0.505448, .5R: 0.000000, .75R: 0.000000, count: 5, class_loss = 2524.311768, iou_loss = 4.568115, total_loss = 2528.879883 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.316946, GIOU: 0.210240), Class: 0.522198, Obj: 0.431191, No Obj: 0.496454, .5R: 0.166667, .75R: 0.000000, count: 6, class_loss = 615.129028, iou_loss = 0.114136, total_loss = 615.243164 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494632, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9719.823242, iou_loss = 0.000000, total_loss = 9719.823242 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.505264, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2521.236328, iou_loss = 0.000000, total_loss = 2521.236328 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.284669, GIOU: 0.234499), Class: 0.447601, Obj: 0.469692, No Obj: 0.495750, .5R: 0.000000, .75R: 0.000000, count: 6, class_loss = 614.089050, iou_loss = 0.075684, total_loss = 614.164734 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.496478, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9794.525391, iou_loss = 0.000000, total_loss = 9794.525391 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.240359, GIOU: 0.222894), Class: 0.425309, Obj: 0.559638, No Obj: 0.505781, .5R: 0.000000, .75R: 0.000000, count: 5, class_loss = 2524.055908, iou_loss = 0.789307, total_loss = 2524.845215 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.373340, GIOU: 0.307888), Class: 0.485156, Obj: 0.459259, No Obj: 0.496429, .5R: 0.250000, .75R: 0.000000, count: 8, class_loss = 616.386597, iou_loss = 0.454407, total_loss = 616.841003 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.218442, GIOU: 0.014144), Class: 0.548967, Obj: 0.555256, No Obj: 0.494791, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 9748.663086, iou_loss = 5.159180, total_loss = 9753.822266 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.331484, GIOU: 0.214746), Class: 0.443538, Obj: 0.470033, No Obj: 0.505419, .5R: 0.214286, .75R: 0.000000, count: 14, class_loss = 2523.048096, iou_loss = 7.005371, total_loss = 2530.053467 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.377443, GIOU: 0.323458), Class: 0.509682, Obj: 0.533418, No Obj: 0.495486, .5R: 0.200000, .75R: 0.100000, count: 10, class_loss = 613.878662, iou_loss = 0.764099, total_loss = 614.642761 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.090361, GIOU: 0.090362), Class: 0.657498, Obj: 0.553246, No Obj: 0.495237, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9736.075195, iou_loss = 0.124023, total_loss = 9736.199219 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.465187, GIOU: 0.447996), Class: 0.420004, Obj: 0.527177, No Obj: 0.505498, .5R: 0.571429, .75R: 0.142857, count: 7, class_loss = 2521.478271, iou_loss = 4.829346, total_loss = 2526.307617 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.351391, GIOU: 0.281612), Class: 0.511184, Obj: 0.518181, No Obj: 0.494662, .5R: 0.055556, .75R: 0.000000, count: 18, class_loss = 613.676270, iou_loss = 0.793152, total_loss = 614.469421 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.495360, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9781.018555, iou_loss = 0.000000, total_loss = 9781.018555 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.205472, GIOU: 0.153024), Class: 0.439573, Obj: 0.641377, No Obj: 0.505327, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 2520.146729, iou_loss = 0.564453, total_loss = 2520.711182 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.456316, GIOU: 0.419444), Class: 0.598590, Obj: 0.420467, No Obj: 0.495351, .5R: 0.500000, .75R: 0.000000, count: 2, class_loss = 613.193542, iou_loss = 0.211365, total_loss = 613.404907 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494419, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9715.309570, iou_loss = 0.000000, total_loss = 9715.309570 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.505200, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2513.228760, iou_loss = 0.000000, total_loss = 2513.228760 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.411856, GIOU: 0.411856), Class: 0.473753, Obj: 0.442685, No Obj: 0.494547, .5R: 0.000000, .75R: 0.000000, count: 2, class_loss = 608.192139, iou_loss = 0.030090, total_loss = 608.222229 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.366005, GIOU: 0.332196), Class: 0.454877, Obj: 0.572724, No Obj: 0.494338, .5R: 0.500000, .75R: 0.000000, count: 2, class_loss = 9710.125977, iou_loss = 3.530273, total_loss = 9713.656250 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.297785, GIOU: 0.134268), Class: 0.483879, Obj: 0.518935, No Obj: 0.505574, .5R: 0.166667, .75R: 0.000000, count: 6, class_loss = 2514.311768, iou_loss = 3.813477, total_loss = 2518.125244 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.336482, GIOU: 0.287838), Class: 0.470483, Obj: 0.491409, No Obj: 0.497424, .5R: 0.157895, .75R: 0.052632, count: 19, class_loss = 618.447632, iou_loss = 0.999756, total_loss = 619.447388 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.177588, GIOU: -0.194522), Class: 0.454748, Obj: 0.547344, No Obj: 0.493574, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9706.048828, iou_loss = 0.146484, total_loss = 9706.195312 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.334370, GIOU: 0.221080), Class: 0.463582, Obj: 0.487536, No Obj: 0.505169, .5R: 0.125000, .75R: 0.000000, count: 16, class_loss = 2521.832275, iou_loss = 8.651123, total_loss = 2530.483398 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.432269, GIOU: 0.399798), Class: 0.487407, Obj: 0.487603, No Obj: 0.495515, .5R: 0.400000, .75R: 0.133333, count: 15, class_loss = 615.902405, iou_loss = 2.327332, total_loss = 618.229736 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.276444, GIOU: 0.268391), Class: 0.496661, Obj: 0.570834, No Obj: 0.494975, .5R: 0.076923, .75R: 0.076923, count: 13, class_loss = 9717.640625, iou_loss = 19.401367, total_loss = 9737.041992 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.359651, GIOU: 0.295835), Class: 0.473917, Obj: 0.535828, No Obj: 0.503798, .5R: 0.086957, .75R: 0.000000, count: 23, class_loss = 2496.943115, iou_loss = 16.721680, total_loss = 2513.664795 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.305970, GIOU: 0.203547), Class: 0.463600, Obj: 0.509077, No Obj: 0.493109, .5R: 0.076923, .75R: 0.000000, count: 13, class_loss = 605.065002, iou_loss = 0.586914, total_loss = 605.651917 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.173589, GIOU: 0.117558), Class: 0.418764, Obj: 0.570068, No Obj: 0.494317, .5R: 0.000000, .75R: 0.000000, count: 4, class_loss = 9703.752930, iou_loss = 1.768555, total_loss = 9705.521484 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.378999, GIOU: 0.307669), Class: 0.482674, Obj: 0.504585, No Obj: 0.504421, .5R: 0.250000, .75R: 0.055556, count: 36, class_loss = 2515.802490, iou_loss = 21.532959, total_loss = 2537.335449 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.385154, GIOU: 0.320011), Class: 0.513996, Obj: 0.502805, No Obj: 0.496023, .5R: 0.241379, .75R: 0.034483, count: 29, class_loss = 618.741211, iou_loss = 3.648010, total_loss = 622.389221 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.314900, GIOU: 0.057883), Class: 0.509073, Obj: 0.509532, No Obj: 0.492939, .5R: 0.142857, .75R: 0.000000, count: 7, class_loss = 9652.234375, iou_loss = 20.753906, total_loss = 9672.988281 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.387441, GIOU: 0.302999), Class: 0.453979, Obj: 0.524787, No Obj: 0.503929, .5R: 0.375000, .75R: 0.000000, count: 16, class_loss = 2500.779541, iou_loss = 7.709961, total_loss = 2508.489502 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.399986, GIOU: 0.336062), Class: 0.477342, Obj: 0.465388, No Obj: 0.495001, .5R: 0.290323, .75R: 0.000000, count: 31, class_loss = 616.839539, iou_loss = 2.705078, total_loss = 619.544617 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.192736, GIOU: -0.017155), Class: 0.375232, Obj: 0.475832, No Obj: 0.494374, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 9702.221680, iou_loss = 2.373047, total_loss = 9704.594727 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.345795, GIOU: 0.192655), Class: 0.552999, Obj: 0.502585, No Obj: 0.504333, .5R: 0.200000, .75R: 0.000000, count: 5, class_loss = 2503.145508, iou_loss = 1.822266, total_loss = 2504.967773 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.498992, GIOU: 0.430461), Class: 0.486338, Obj: 0.437027, No Obj: 0.495801, .5R: 0.428571, .75R: 0.285714, count: 7, class_loss = 612.831543, iou_loss = 0.220642, total_loss = 613.052185 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.495194, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9754.033203, iou_loss = 0.000000, total_loss = 9754.033203 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.279229, GIOU: 0.230572), Class: 0.430075, Obj: 0.491835, No Obj: 0.505289, .5R: 0.000000, .75R: 0.000000, count: 4, class_loss = 2523.378174, iou_loss = 0.879883, total_loss = 2524.258057 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.383953, GIOU: 0.274982), Class: 0.549446, Obj: 0.464767, No Obj: 0.495310, .5R: 0.285714, .75R: 0.000000, count: 7, class_loss = 613.210205, iou_loss = 0.406372, total_loss = 613.616577 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.496466, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9810.359375, iou_loss = 0.000000, total_loss = 9810.359375 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.463904, GIOU: 0.394583), Class: 0.491594, Obj: 0.521811, No Obj: 0.504006, .5R: 0.333333, .75R: 0.166667, count: 6, class_loss = 2503.735596, iou_loss = 3.940674, total_loss = 2507.676270 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.450772, GIOU: 0.404599), Class: 0.613081, Obj: 0.542935, No Obj: 0.496232, .5R: 0.500000, .75R: 0.000000, count: 4, class_loss = 613.129822, iou_loss = 1.014038, total_loss = 614.143860 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494728, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9710.630859, iou_loss = 0.000000, total_loss = 9710.630859 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.297150, GIOU: 0.228874), Class: 0.488531, Obj: 0.501682, No Obj: 0.504182, .5R: 0.000000, .75R: 0.000000, count: 9, class_loss = 2506.686523, iou_loss = 3.297852, total_loss = 2509.984375 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.475856, GIOU: 0.466200), Class: 0.434663, Obj: 0.550578, No Obj: 0.496101, .5R: 0.400000, .75R: 0.200000, count: 5, class_loss = 613.884521, iou_loss = 2.105408, total_loss = 615.989929 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.493643, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9687.576172, iou_loss = 0.000000, total_loss = 9687.576172 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.504768, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2511.071045, iou_loss = 0.000000, total_loss = 2511.071045 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.490553, GIOU: 0.485102), Class: 0.471723, Obj: 0.606182, No Obj: 0.496635, .5R: 0.333333, .75R: 0.000000, count: 3, class_loss = 614.543030, iou_loss = 0.115662, total_loss = 614.658691 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.198209, GIOU: 0.010955), Class: 0.497410, Obj: 0.480301, No Obj: 0.494223, .5R: 0.000000, .75R: 0.000000, count: 7, class_loss = 9689.848633, iou_loss = 3.983398, total_loss = 9693.832031 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.304636, GIOU: 0.161018), Class: 0.477203, Obj: 0.479711, No Obj: 0.504729, .5R: 0.166667, .75R: 0.000000, count: 36, class_loss = 2518.990479, iou_loss = 18.632080, total_loss = 2537.622559 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.280161, GIOU: 0.095836), Class: 0.541028, Obj: 0.543954, No Obj: 0.495875, .5R: 0.000000, .75R: 0.000000, count: 16, class_loss = 613.112610, iou_loss = 0.791016, total_loss = 613.903625 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.191343, GIOU: 0.092096), Class: 0.413619, Obj: 0.427680, No Obj: 0.495591, .5R: 0.000000, .75R: 0.000000, count: 4, class_loss = 9783.933594, iou_loss = 11.540039, total_loss = 9795.473633 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.416392, GIOU: 0.358835), Class: 0.456800, Obj: 0.505720, No Obj: 0.505963, .5R: 0.222222, .75R: 0.055556, count: 18, class_loss = 2533.290771, iou_loss = 9.486328, total_loss = 2542.777100 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.363643, GIOU: 0.320993), Class: 0.502116, Obj: 0.500705, No Obj: 0.496789, .5R: 0.090909, .75R: 0.000000, count: 11, class_loss = 618.275085, iou_loss = 1.030396, total_loss = 619.305481 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494697, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9758.993164, iou_loss = 0.000000, total_loss = 9758.993164 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.220871, GIOU: 0.168125), Class: 0.297975, Obj: 0.507962, No Obj: 0.504385, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 2510.186523, iou_loss = 0.309326, total_loss = 2510.495850 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.274232, GIOU: 0.123809), Class: 0.537184, Obj: 0.458978, No Obj: 0.494412, .5R: 0.000000, .75R: 0.000000, count: 7, class_loss = 611.082336, iou_loss = 0.169006, total_loss = 611.251343 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.237861, GIOU: 0.044680), Class: 0.596454, Obj: 0.504906, No Obj: 0.494371, .5R: 0.166667, .75R: 0.000000, count: 6, class_loss = 9725.442383, iou_loss = 27.629883, total_loss = 9753.072266 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.362619, GIOU: 0.354029), Class: 0.489653, Obj: 0.477481, No Obj: 0.506308, .5R: 0.250000, .75R: 0.000000, count: 4, class_loss = 2534.077148, iou_loss = 1.637451, total_loss = 2535.714600 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.400123, GIOU: 0.330602), Class: 0.552369, Obj: 0.473562, No Obj: 0.494837, .5R: 0.230769, .75R: 0.076923, count: 13, class_loss = 613.371338, iou_loss = 0.479370, total_loss = 613.850708 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.143174, GIOU: 0.143174), Class: 0.551383, Obj: 0.465031, No Obj: 0.495072, .5R: 0.000000, .75R: 0.000000, count: 2, class_loss = 9756.158203, iou_loss = 0.805664, total_loss = 9756.963867 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.426819, GIOU: 0.376955), Class: 0.477330, Obj: 0.504982, No Obj: 0.505291, .5R: 0.333333, .75R: 0.083333, count: 12, class_loss = 2524.931885, iou_loss = 9.124023, total_loss = 2534.055908 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.411569, GIOU: 0.335957), Class: 0.507815, Obj: 0.434154, No Obj: 0.495928, .5R: 0.250000, .75R: 0.062500, count: 16, class_loss = 618.675842, iou_loss = 1.158569, total_loss = 619.834412 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.493776, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9675.855469, iou_loss = 0.000000, total_loss = 9675.855469 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.504400, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2504.437500, iou_loss = 0.000000, total_loss = 2504.437500 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.403511, GIOU: 0.238823), Class: 0.433387, Obj: 0.390465, No Obj: 0.496106, .5R: 0.500000, .75R: 0.000000, count: 2, class_loss = 612.543152, iou_loss = 0.027100, total_loss = 612.570251 
    
     Tensor Cores are disabled until the first 3000 iterations are reached.
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.271456, GIOU: 0.098098), Class: 0.583218, Obj: 0.526473, No Obj: 0.493374, .5R: 0.043478, .75R: 0.000000, count: 23, class_loss = 9684.161133, iou_loss = 52.929688, total_loss = 9737.090820 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.315942, GIOU: 0.263115), Class: 0.541790, Obj: 0.519851, No Obj: 0.504433, .5R: 0.133333, .75R: 0.000000, count: 15, class_loss = 2507.070557, iou_loss = 13.455566, total_loss = 2520.526123 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.387588, GIOU: 0.356964), Class: 0.499589, Obj: 0.447929, No Obj: 0.496652, .5R: 0.333333, .75R: 0.000000, count: 6, class_loss = 614.430786, iou_loss = 0.307190, total_loss = 614.737976 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.493891, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9679.644531, iou_loss = 0.000000, total_loss = 9679.644531 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.308803, GIOU: 0.272873), Class: 0.485631, Obj: 0.462466, No Obj: 0.504408, .5R: 0.250000, .75R: 0.000000, count: 4, class_loss = 2504.601807, iou_loss = 1.521729, total_loss = 2506.123535 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.307402, GIOU: 0.270846), Class: 0.486569, Obj: 0.480870, No Obj: 0.495372, .5R: 0.066667, .75R: 0.000000, count: 15, class_loss = 613.220276, iou_loss = 0.489319, total_loss = 613.709595 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.250383, GIOU: 0.126767), Class: 0.585468, Obj: 0.458561, No Obj: 0.493028, .5R: 0.117647, .75R: 0.000000, count: 17, class_loss = 9688.948242, iou_loss = 46.924805, total_loss = 9735.873047 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.475709, GIOU: 0.424249), Class: 0.469633, Obj: 0.491398, No Obj: 0.505935, .5R: 0.500000, .75R: 0.100000, count: 20, class_loss = 2528.628662, iou_loss = 19.636719, total_loss = 2548.265381 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.380185, GIOU: 0.347352), Class: 0.505630, Obj: 0.519510, No Obj: 0.498106, .5R: 0.210526, .75R: 0.000000, count: 19, class_loss = 622.158020, iou_loss = 1.957703, total_loss = 624.115723 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.386447, GIOU: 0.278767), Class: 0.433297, Obj: 0.515626, No Obj: 0.494014, .5R: 0.333333, .75R: 0.166667, count: 6, class_loss = 9690.596680, iou_loss = 17.526367, total_loss = 9708.123047 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.337697, GIOU: 0.265325), Class: 0.454289, Obj: 0.532947, No Obj: 0.505287, .5R: 0.142857, .75R: 0.000000, count: 14, class_loss = 2515.488037, iou_loss = 6.147461, total_loss = 2521.635498 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.368785, GIOU: 0.334681), Class: 0.514290, Obj: 0.507571, No Obj: 0.496408, .5R: 0.300000, .75R: 0.000000, count: 20, class_loss = 614.957947, iou_loss = 1.199768, total_loss = 616.157715 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.291731, GIOU: 0.237486), Class: 0.522401, Obj: 0.678651, No Obj: 0.493875, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9706.560547, iou_loss = 5.000977, total_loss = 9711.561523 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.278220, GIOU: 0.127210), Class: 0.458007, Obj: 0.483230, No Obj: 0.505363, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 2513.622559, iou_loss = 0.300293, total_loss = 2513.922852 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.362419, GIOU: 0.273984), Class: 0.503653, Obj: 0.496498, No Obj: 0.495077, .5R: 0.263158, .75R: 0.052632, count: 19, class_loss = 612.737976, iou_loss = 1.020752, total_loss = 613.758728 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.165500, GIOU: -0.326142), Class: 0.589067, Obj: 0.369113, No Obj: 0.494197, .5R: 0.000000, .75R: 0.000000, count: 2, class_loss = 9692.134766, iou_loss = 1.066406, total_loss = 9693.201172 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.359169, GIOU: 0.290521), Class: 0.468112, Obj: 0.520981, No Obj: 0.504331, .5R: 0.100000, .75R: 0.000000, count: 10, class_loss = 2503.126221, iou_loss = 4.987549, total_loss = 2508.113770 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.377061, GIOU: 0.295372), Class: 0.509787, Obj: 0.469943, No Obj: 0.496164, .5R: 0.208333, .75R: 0.083333, count: 24, class_loss = 616.811951, iou_loss = 1.118469, total_loss = 617.930420 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.257972, GIOU: 0.106257), Class: 0.555307, Obj: 0.512833, No Obj: 0.494343, .5R: 0.041667, .75R: 0.000000, count: 24, class_loss = 9696.687500, iou_loss = 39.726562, total_loss = 9736.414062 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.362063, GIOU: 0.276042), Class: 0.491589, Obj: 0.521329, No Obj: 0.504739, .5R: 0.153846, .75R: 0.025641, count: 39, class_loss = 2513.856201, iou_loss = 21.413818, total_loss = 2535.270020 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.359948, GIOU: 0.303811), Class: 0.511649, Obj: 0.500525, No Obj: 0.497355, .5R: 0.157895, .75R: 0.000000, count: 19, class_loss = 617.527100, iou_loss = 2.237610, total_loss = 619.764709 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.212893, GIOU: -0.076586), Class: 0.487597, Obj: 0.479586, No Obj: 0.494540, .5R: 0.000000, .75R: 0.000000, count: 13, class_loss = 9717.355469, iou_loss = 10.268555, total_loss = 9727.624023 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.337551, GIOU: 0.179038), Class: 0.521516, Obj: 0.510383, No Obj: 0.505069, .5R: 0.241379, .75R: 0.103448, count: 29, class_loss = 2518.798340, iou_loss = 20.720215, total_loss = 2539.518555 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.345708, GIOU: 0.247562), Class: 0.501029, Obj: 0.521624, No Obj: 0.495274, .5R: 0.083333, .75R: 0.000000, count: 12, class_loss = 612.505615, iou_loss = 0.689392, total_loss = 613.195007 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494059, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9705.174805, iou_loss = 0.000000, total_loss = 9705.174805 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.417114, GIOU: 0.272054), Class: 0.474566, Obj: 0.560565, No Obj: 0.505117, .5R: 0.666667, .75R: 0.000000, count: 3, class_loss = 2510.133057, iou_loss = 2.932861, total_loss = 2513.065918 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.426126, GIOU: 0.372131), Class: 0.496900, Obj: 0.513787, No Obj: 0.496108, .5R: 0.466667, .75R: 0.066667, count: 15, class_loss = 613.322327, iou_loss = 0.857483, total_loss = 614.179810 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494369, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9709.675781, iou_loss = 0.000000, total_loss = 9709.675781 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.267764, GIOU: 0.071932), Class: 0.441279, Obj: 0.502924, No Obj: 0.504977, .5R: 0.000000, .75R: 0.000000, count: 11, class_loss = 2515.639648, iou_loss = 2.127197, total_loss = 2517.766846 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.371509, GIOU: 0.253348), Class: 0.482441, Obj: 0.512823, No Obj: 0.496553, .5R: 0.055556, .75R: 0.000000, count: 18, class_loss = 617.659119, iou_loss = 1.672546, total_loss = 619.331665 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494348, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9689.464844, iou_loss = 0.000000, total_loss = 9689.464844 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.282391, GIOU: 0.132089), Class: 0.419362, Obj: 0.441562, No Obj: 0.505665, .5R: 0.000000, .75R: 0.000000, count: 5, class_loss = 2516.482910, iou_loss = 1.168701, total_loss = 2517.651611 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.315350, GIOU: 0.235771), Class: 0.517178, Obj: 0.493191, No Obj: 0.496644, .5R: 0.115385, .75R: 0.038462, count: 26, class_loss = 617.888550, iou_loss = 1.039734, total_loss = 618.928284 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.495614, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9763.501953, iou_loss = 0.000000, total_loss = 9763.501953 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.315008, GIOU: 0.314352), Class: 0.448194, Obj: 0.539412, No Obj: 0.504673, .5R: 0.000000, .75R: 0.000000, count: 4, class_loss = 2518.187012, iou_loss = 0.697998, total_loss = 2518.885010 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.437363, GIOU: 0.415179), Class: 0.498491, Obj: 0.515408, No Obj: 0.494541, .5R: 0.333333, .75R: 0.055556, count: 18, class_loss = 613.979187, iou_loss = 2.223938, total_loss = 616.203125 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494737, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9735.962891, iou_loss = 0.000000, total_loss = 9735.962891 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.396867, GIOU: 0.396118), Class: 0.349462, Obj: 0.466238, No Obj: 0.505166, .5R: 0.500000, .75R: 0.000000, count: 2, class_loss = 2521.454102, iou_loss = 0.921143, total_loss = 2522.375244 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.329645, GIOU: 0.189591), Class: 0.541660, Obj: 0.508278, No Obj: 0.495693, .5R: 0.333333, .75R: 0.000000, count: 9, class_loss = 614.534973, iou_loss = 0.821289, total_loss = 615.356262 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.495146, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9744.178711, iou_loss = 0.000000, total_loss = 9744.178711 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.504986, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2507.027344, iou_loss = 0.000000, total_loss = 2507.027344 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.506713, GIOU: 0.486540), Class: 0.592651, Obj: 0.417398, No Obj: 0.494999, .5R: 0.250000, .75R: 0.250000, count: 4, class_loss = 610.149353, iou_loss = 0.177246, total_loss = 610.326599 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.179875, GIOU: -0.062856), Class: 0.453208, Obj: 0.344160, No Obj: 0.494319, .5R: 0.000000, .75R: 0.000000, count: 4, class_loss = 9693.670898, iou_loss = 1.879883, total_loss = 9695.550781 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.261187, GIOU: 0.131867), Class: 0.405456, Obj: 0.497804, No Obj: 0.505027, .5R: 0.125000, .75R: 0.000000, count: 16, class_loss = 2513.326172, iou_loss = 10.964355, total_loss = 2524.290527 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.270044, GIOU: 0.086502), Class: 0.534766, Obj: 0.478566, No Obj: 0.496052, .5R: 0.000000, .75R: 0.000000, count: 11, class_loss = 613.187134, iou_loss = 0.291809, total_loss = 613.478943 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.495531, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9790.024414, iou_loss = 0.000000, total_loss = 9790.024414 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.462409, GIOU: 0.414551), Class: 0.526785, Obj: 0.525847, No Obj: 0.505112, .5R: 0.250000, .75R: 0.000000, count: 4, class_loss = 2516.453613, iou_loss = 2.815186, total_loss = 2519.268799 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.485223, GIOU: 0.482694), Class: 0.503458, Obj: 0.493045, No Obj: 0.495349, .5R: 0.666667, .75R: 0.000000, count: 9, class_loss = 613.523560, iou_loss = 1.113464, total_loss = 614.637024 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.185027, GIOU: 0.084102), Class: 0.466025, Obj: 0.566547, No Obj: 0.495408, .5R: 0.000000, .75R: 0.000000, count: 5, class_loss = 9773.162109, iou_loss = 3.319336, total_loss = 9776.481445 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.391605, GIOU: 0.332954), Class: 0.520850, Obj: 0.465558, No Obj: 0.505237, .5R: 0.250000, .75R: 0.083333, count: 12, class_loss = 2520.737305, iou_loss = 6.370117, total_loss = 2527.107422 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.315055, GIOU: 0.184859), Class: 0.505286, Obj: 0.518074, No Obj: 0.494973, .5R: 0.000000, .75R: 0.000000, count: 13, class_loss = 612.660339, iou_loss = 0.586365, total_loss = 613.246704 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.495946, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9790.145508, iou_loss = 0.000000, total_loss = 9790.145508 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.455594, GIOU: 0.450926), Class: 0.497522, Obj: 0.537961, No Obj: 0.504485, .5R: 0.428571, .75R: 0.000000, count: 7, class_loss = 2509.614746, iou_loss = 8.345459, total_loss = 2517.960205 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.297881, GIOU: 0.197994), Class: 0.517378, Obj: 0.482685, No Obj: 0.497261, .5R: 0.000000, .75R: 0.000000, count: 5, class_loss = 617.779114, iou_loss = 0.310608, total_loss = 618.089722 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.231002, GIOU: 0.097929), Class: 0.471224, Obj: 0.524424, No Obj: 0.493907, .5R: 0.000000, .75R: 0.000000, count: 4, class_loss = 9688.355469, iou_loss = 5.442383, total_loss = 9693.797852 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.289973, GIOU: 0.232110), Class: 0.466902, Obj: 0.474736, No Obj: 0.505264, .5R: 0.187500, .75R: 0.000000, count: 16, class_loss = 2520.383545, iou_loss = 3.981689, total_loss = 2524.365234 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.369207, GIOU: 0.317743), Class: 0.500787, Obj: 0.517052, No Obj: 0.495591, .5R: 0.090909, .75R: 0.000000, count: 11, class_loss = 612.562622, iou_loss = 1.797729, total_loss = 614.360352 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494984, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9758.010742, iou_loss = 0.000000, total_loss = 9758.010742 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.332236, GIOU: 0.263436), Class: 0.406775, Obj: 0.533175, No Obj: 0.505454, .5R: 0.333333, .75R: 0.000000, count: 3, class_loss = 2525.088623, iou_loss = 0.599609, total_loss = 2525.688232 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.413606, GIOU: 0.369340), Class: 0.553593, Obj: 0.514922, No Obj: 0.496199, .5R: 0.200000, .75R: 0.000000, count: 5, class_loss = 614.472900, iou_loss = 0.294556, total_loss = 614.767456 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.495274, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9758.884766, iou_loss = 0.000000, total_loss = 9758.884766 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.199264, GIOU: 0.115680), Class: 0.492132, Obj: 0.453088, No Obj: 0.505560, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 2523.547852, iou_loss = 0.314453, total_loss = 2523.862305 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.330672, GIOU: 0.207574), Class: 0.476825, Obj: 0.453750, No Obj: 0.495107, .5R: 0.111111, .75R: 0.000000, count: 9, class_loss = 614.353760, iou_loss = 0.472290, total_loss = 614.826050 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.493946, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9714.352539, iou_loss = 0.000000, total_loss = 9714.352539 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.526619, GIOU: 0.526619), Class: 0.369150, Obj: 0.595766, No Obj: 0.504931, .5R: 1.000000, .75R: 0.000000, count: 1, class_loss = 2516.752441, iou_loss = 0.523438, total_loss = 2517.275879 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.347182, GIOU: 0.281366), Class: 0.497322, Obj: 0.480148, No Obj: 0.496522, .5R: 0.176471, .75R: 0.000000, count: 17, class_loss = 618.803467, iou_loss = 0.803589, total_loss = 619.607056 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.257511, GIOU: 0.035425), Class: 0.680128, Obj: 0.385153, No Obj: 0.494041, .5R: 0.125000, .75R: 0.000000, count: 16, class_loss = 9702.698242, iou_loss = 32.482422, total_loss = 9735.180664 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.325919, GIOU: 0.322304), Class: 0.444378, Obj: 0.524945, No Obj: 0.504726, .5R: 0.000000, .75R: 0.000000, count: 8, class_loss = 2510.480713, iou_loss = 4.912109, total_loss = 2515.392822 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.369686, GIOU: 0.355469), Class: 0.565428, Obj: 0.485679, No Obj: 0.496269, .5R: 0.375000, .75R: 0.000000, count: 8, class_loss = 613.944702, iou_loss = 0.378479, total_loss = 614.323181 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.127239, GIOU: -0.055454), Class: 0.397009, Obj: 0.519533, No Obj: 0.493717, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9705.965820, iou_loss = 0.292969, total_loss = 9706.258789 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.240387, GIOU: 0.057756), Class: 0.412717, Obj: 0.489931, No Obj: 0.505915, .5R: 0.000000, .75R: 0.000000, count: 11, class_loss = 2523.844482, iou_loss = 2.090576, total_loss = 2525.935059 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.494711, GIOU: 0.452624), Class: 0.454889, Obj: 0.502721, No Obj: 0.496864, .5R: 0.545455, .75R: 0.181818, count: 11, class_loss = 615.792175, iou_loss = 1.308289, total_loss = 617.100464 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.089716, GIOU: -0.258522), Class: 0.343151, Obj: 0.372265, No Obj: 0.494145, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9690.705078, iou_loss = 0.048828, total_loss = 9690.753906 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.340389, GIOU: 0.224213), Class: 0.445319, Obj: 0.521776, No Obj: 0.504358, .5R: 0.090909, .75R: 0.090909, count: 11, class_loss = 2506.013428, iou_loss = 3.800781, total_loss = 2509.814209 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.385819, GIOU: 0.339426), Class: 0.526398, Obj: 0.463045, No Obj: 0.496475, .5R: 0.363636, .75R: 0.045455, count: 22, class_loss = 618.291870, iou_loss = 1.348083, total_loss = 619.639954 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.258305, GIOU: 0.075645), Class: 0.548620, Obj: 0.462454, No Obj: 0.494146, .5R: 0.125000, .75R: 0.000000, count: 32, class_loss = 9722.146484, iou_loss = 63.140625, total_loss = 9785.287109 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.354746, GIOU: 0.284631), Class: 0.506667, Obj: 0.515540, No Obj: 0.504306, .5R: 0.175000, .75R: 0.050000, count: 40, class_loss = 2509.906006, iou_loss = 30.211182, total_loss = 2540.117188 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.379355, GIOU: 0.307071), Class: 0.528079, Obj: 0.459841, No Obj: 0.496712, .5R: 0.300000, .75R: 0.050000, count: 20, class_loss = 617.241455, iou_loss = 3.149658, total_loss = 620.391113 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.233576, GIOU: 0.189060), Class: 0.497325, Obj: 0.546129, No Obj: 0.494450, .5R: 0.000000, .75R: 0.000000, count: 7, class_loss = 9715.793945, iou_loss = 8.567383, total_loss = 9724.361328 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.372575, GIOU: 0.263363), Class: 0.467144, Obj: 0.498988, No Obj: 0.504411, .5R: 0.240000, .75R: 0.000000, count: 25, class_loss = 2517.872314, iou_loss = 14.321533, total_loss = 2532.193848 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.338797, GIOU: 0.255392), Class: 0.494339, Obj: 0.493809, No Obj: 0.495816, .5R: 0.166667, .75R: 0.041667, count: 24, class_loss = 616.374146, iou_loss = 1.782776, total_loss = 618.156921 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.263715, GIOU: 0.014255), Class: 0.733107, Obj: 0.376180, No Obj: 0.493917, .5R: 0.000000, .75R: 0.000000, count: 4, class_loss = 9662.964844, iou_loss = 22.801758, total_loss = 9685.766602 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.330134, GIOU: 0.244775), Class: 0.475929, Obj: 0.424553, No Obj: 0.504459, .5R: 0.000000, .75R: 0.000000, count: 5, class_loss = 2505.504150, iou_loss = 0.972656, total_loss = 2506.476807 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.382347, GIOU: 0.320396), Class: 0.491340, Obj: 0.512482, No Obj: 0.496697, .5R: 0.235294, .75R: 0.058824, count: 17, class_loss = 616.101746, iou_loss = 0.667603, total_loss = 616.769348 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.148855, GIOU: -0.291570), Class: 0.453027, Obj: 0.407418, No Obj: 0.493519, .5R: 0.000000, .75R: 0.000000, count: 2, class_loss = 9658.949219, iou_loss = 9.800781, total_loss = 9668.750000 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.313324, GIOU: 0.234593), Class: 0.474244, Obj: 0.482237, No Obj: 0.504385, .5R: 0.137931, .75R: 0.000000, count: 29, class_loss = 2511.444580, iou_loss = 10.423828, total_loss = 2521.868408 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.358062, GIOU: 0.279862), Class: 0.532227, Obj: 0.488438, No Obj: 0.496067, .5R: 0.192308, .75R: 0.000000, count: 26, class_loss = 614.427612, iou_loss = 1.904114, total_loss = 616.331726 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.232530, GIOU: -0.040479), Class: 0.593739, Obj: 0.393295, No Obj: 0.494478, .5R: 0.045455, .75R: 0.000000, count: 22, class_loss = 9751.424805, iou_loss = 68.373047, total_loss = 9819.797852 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.277402, GIOU: 0.146302), Class: 0.495446, Obj: 0.540433, No Obj: 0.504987, .5R: 0.090909, .75R: 0.000000, count: 11, class_loss = 2512.183838, iou_loss = 3.200439, total_loss = 2515.384277 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.327065, GIOU: 0.245852), Class: 0.496273, Obj: 0.485812, No Obj: 0.496805, .5R: 0.192308, .75R: 0.000000, count: 26, class_loss = 619.787415, iou_loss = 1.432678, total_loss = 621.220093 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.296805, GIOU: 0.111627), Class: 0.510999, Obj: 0.493971, No Obj: 0.493837, .5R: 0.000000, .75R: 0.000000, count: 7, class_loss = 9700.183594, iou_loss = 10.976562, total_loss = 9711.160156 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.329521, GIOU: 0.201943), Class: 0.461614, Obj: 0.465076, No Obj: 0.503940, .5R: 0.064516, .75R: 0.000000, count: 31, class_loss = 2508.389404, iou_loss = 17.694092, total_loss = 2526.083496 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.382712, GIOU: 0.311025), Class: 0.521795, Obj: 0.495288, No Obj: 0.496820, .5R: 0.259259, .75R: 0.037037, count: 27, class_loss = 620.186218, iou_loss = 4.882568, total_loss = 625.068787 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.414149, GIOU: 0.413613), Class: 0.531392, Obj: 0.465112, No Obj: 0.494115, .5R: 0.333333, .75R: 0.333333, count: 3, class_loss = 9714.675781, iou_loss = 16.975586, total_loss = 9731.651367 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.368261, GIOU: 0.298123), Class: 0.429104, Obj: 0.455436, No Obj: 0.504725, .5R: 0.227273, .75R: 0.000000, count: 22, class_loss = 2519.307861, iou_loss = 10.497803, total_loss = 2529.805664 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.384949, GIOU: 0.305438), Class: 0.527741, Obj: 0.482318, No Obj: 0.496432, .5R: 0.208333, .75R: 0.041667, count: 24, class_loss = 618.488953, iou_loss = 1.636414, total_loss = 620.125366 
    
     Tensor Cores are disabled until the first 3000 iterations are reached.
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.256317, GIOU: 0.062522), Class: 0.523009, Obj: 0.487388, No Obj: 0.493712, .5R: 0.066667, .75R: 0.000000, count: 15, class_loss = 9667.128906, iou_loss = 37.182617, total_loss = 9704.311523 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.363232, GIOU: 0.252969), Class: 0.513402, Obj: 0.509457, No Obj: 0.504197, .5R: 0.200000, .75R: 0.000000, count: 25, class_loss = 2507.146240, iou_loss = 19.866943, total_loss = 2527.013184 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.433415, GIOU: 0.339479), Class: 0.490062, Obj: 0.535814, No Obj: 0.496673, .5R: 0.363636, .75R: 0.045455, count: 22, class_loss = 614.828430, iou_loss = 1.555481, total_loss = 616.383911 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.195568, GIOU: -0.071919), Class: 0.694338, Obj: 0.528452, No Obj: 0.493630, .5R: 0.000000, .75R: 0.000000, count: 2, class_loss = 9678.591797, iou_loss = 1.618164, total_loss = 9680.209961 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.340249, GIOU: 0.284206), Class: 0.474250, Obj: 0.482419, No Obj: 0.504878, .5R: 0.166667, .75R: 0.083333, count: 12, class_loss = 2511.541504, iou_loss = 5.739258, total_loss = 2517.280762 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.384125, GIOU: 0.296209), Class: 0.504320, Obj: 0.499019, No Obj: 0.496282, .5R: 0.240000, .75R: 0.000000, count: 25, class_loss = 617.551086, iou_loss = 1.974976, total_loss = 619.526062 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.275747, GIOU: 0.130356), Class: 0.539949, Obj: 0.448023, No Obj: 0.494218, .5R: 0.235294, .75R: 0.000000, count: 17, class_loss = 9716.828125, iou_loss = 52.591797, total_loss = 9769.419922 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.319147, GIOU: 0.238242), Class: 0.474782, Obj: 0.512273, No Obj: 0.503554, .5R: 0.173913, .75R: 0.000000, count: 23, class_loss = 2505.019775, iou_loss = 12.690674, total_loss = 2517.710449 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.399976, GIOU: 0.321295), Class: 0.495740, Obj: 0.477216, No Obj: 0.494103, .5R: 0.333333, .75R: 0.041667, count: 24, class_loss = 613.725464, iou_loss = 2.110657, total_loss = 615.836121 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.027694, GIOU: -0.867383), Class: 0.298641, Obj: 0.498374, No Obj: 0.493762, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9694.917969, iou_loss = 0.003906, total_loss = 9694.921875 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.336545, GIOU: 0.222960), Class: 0.454143, Obj: 0.609635, No Obj: 0.505195, .5R: 0.333333, .75R: 0.333333, count: 3, class_loss = 2519.340820, iou_loss = 1.524902, total_loss = 2520.865723 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.393284, GIOU: 0.311151), Class: 0.474785, Obj: 0.483795, No Obj: 0.496468, .5R: 0.277778, .75R: 0.111111, count: 18, class_loss = 617.874451, iou_loss = 1.273010, total_loss = 619.147461 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.104557, GIOU: -0.550988), Class: 0.588081, Obj: 0.439813, No Obj: 0.493674, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9701.163086, iou_loss = 0.055664, total_loss = 9701.218750 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.321120, GIOU: 0.195181), Class: 0.461669, Obj: 0.458179, No Obj: 0.504976, .5R: 0.222222, .75R: 0.000000, count: 9, class_loss = 2513.909912, iou_loss = 3.433594, total_loss = 2517.343506 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.310028, GIOU: 0.210678), Class: 0.526727, Obj: 0.449627, No Obj: 0.496722, .5R: 0.000000, .75R: 0.000000, count: 12, class_loss = 615.656189, iou_loss = 0.563843, total_loss = 616.220032 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.355467, GIOU: 0.332733), Class: 0.468804, Obj: 0.585035, No Obj: 0.494230, .5R: 0.285714, .75R: 0.000000, count: 7, class_loss = 9706.569336, iou_loss = 31.713867, total_loss = 9738.283203 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.357351, GIOU: 0.270440), Class: 0.427791, Obj: 0.525202, No Obj: 0.504986, .5R: 0.222222, .75R: 0.027778, count: 36, class_loss = 2522.474854, iou_loss = 15.968994, total_loss = 2538.443848 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.388263, GIOU: 0.304646), Class: 0.517081, Obj: 0.483632, No Obj: 0.495977, .5R: 0.388889, .75R: 0.000000, count: 36, class_loss = 622.171631, iou_loss = 4.168457, total_loss = 626.340088 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.164192, GIOU: -0.029853), Class: 0.551357, Obj: 0.670666, No Obj: 0.494912, .5R: 0.000000, .75R: 0.000000, count: 2, class_loss = 9727.425781, iou_loss = 0.649414, total_loss = 9728.075195 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.389372, GIOU: 0.287230), Class: 0.471728, Obj: 0.525443, No Obj: 0.505438, .5R: 0.111111, .75R: 0.000000, count: 9, class_loss = 2519.049072, iou_loss = 6.471680, total_loss = 2525.520752 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.379615, GIOU: 0.311714), Class: 0.468595, Obj: 0.495342, No Obj: 0.495479, .5R: 0.150000, .75R: 0.000000, count: 20, class_loss = 615.587158, iou_loss = 0.967712, total_loss = 616.554871 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.280357, GIOU: 0.211079), Class: 0.545696, Obj: 0.575567, No Obj: 0.494024, .5R: 0.083333, .75R: 0.000000, count: 12, class_loss = 9702.028320, iou_loss = 18.867188, total_loss = 9720.895508 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.355423, GIOU: 0.265223), Class: 0.494174, Obj: 0.552205, No Obj: 0.505231, .5R: 0.269231, .75R: 0.000000, count: 26, class_loss = 2515.351807, iou_loss = 12.896484, total_loss = 2528.248291 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.342919, GIOU: 0.250958), Class: 0.484999, Obj: 0.502388, No Obj: 0.494400, .5R: 0.307692, .75R: 0.000000, count: 26, class_loss = 613.927002, iou_loss = 1.790100, total_loss = 615.717102 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.492881, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9659.039062, iou_loss = 0.000000, total_loss = 9659.039062 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.434312, GIOU: 0.394459), Class: 0.446335, Obj: 0.481633, No Obj: 0.505172, .5R: 0.250000, .75R: 0.125000, count: 8, class_loss = 2515.486572, iou_loss = 4.490234, total_loss = 2519.976807 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.357834, GIOU: 0.303020), Class: 0.469972, Obj: 0.508657, No Obj: 0.496374, .5R: 0.315789, .75R: 0.000000, count: 19, class_loss = 617.377502, iou_loss = 1.929688, total_loss = 619.307190 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.391427, GIOU: 0.277572), Class: 0.565439, Obj: 0.588767, No Obj: 0.493476, .5R: 0.333333, .75R: 0.000000, count: 3, class_loss = 9658.045898, iou_loss = 19.830078, total_loss = 9677.875977 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.387199, GIOU: 0.331703), Class: 0.409526, Obj: 0.504750, No Obj: 0.504483, .5R: 0.142857, .75R: 0.000000, count: 7, class_loss = 2505.773682, iou_loss = 3.718750, total_loss = 2509.492432 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.401541, GIOU: 0.334445), Class: 0.502466, Obj: 0.470112, No Obj: 0.497717, .5R: 0.294118, .75R: 0.000000, count: 17, class_loss = 618.927734, iou_loss = 1.002441, total_loss = 619.930176 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.273237, GIOU: 0.169356), Class: 0.544618, Obj: 0.461859, No Obj: 0.493654, .5R: 0.142857, .75R: 0.000000, count: 7, class_loss = 9676.602539, iou_loss = 11.286133, total_loss = 9687.888672 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.351066, GIOU: 0.237213), Class: 0.565864, Obj: 0.527571, No Obj: 0.504570, .5R: 0.200000, .75R: 0.000000, count: 15, class_loss = 2511.505127, iou_loss = 11.926758, total_loss = 2523.431885 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.303206, GIOU: 0.248441), Class: 0.500056, Obj: 0.445595, No Obj: 0.495704, .5R: 0.111111, .75R: 0.000000, count: 18, class_loss = 614.792847, iou_loss = 0.546814, total_loss = 615.339661 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.493639, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9685.225586, iou_loss = 0.000000, total_loss = 9685.225586 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.377724, GIOU: 0.194910), Class: 0.443072, Obj: 0.462459, No Obj: 0.504976, .5R: 0.222222, .75R: 0.111111, count: 9, class_loss = 2514.243652, iou_loss = 3.562744, total_loss = 2517.806396 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.467694, GIOU: 0.419408), Class: 0.519106, Obj: 0.484260, No Obj: 0.496509, .5R: 0.523810, .75R: 0.047619, count: 21, class_loss = 617.946228, iou_loss = 2.224304, total_loss = 620.170532 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.211115, GIOU: 0.035888), Class: 0.549763, Obj: 0.548904, No Obj: 0.494796, .5R: 0.000000, .75R: 0.000000, count: 7, class_loss = 9732.611328, iou_loss = 5.446289, total_loss = 9738.057617 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.320883, GIOU: 0.220492), Class: 0.449711, Obj: 0.527881, No Obj: 0.504879, .5R: 0.076923, .75R: 0.000000, count: 26, class_loss = 2513.991455, iou_loss = 17.682861, total_loss = 2531.674316 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.430809, GIOU: 0.367531), Class: 0.503355, Obj: 0.480998, No Obj: 0.496831, .5R: 0.388889, .75R: 0.111111, count: 18, class_loss = 616.236633, iou_loss = 1.694275, total_loss = 617.930908 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.320295, GIOU: 0.182378), Class: 0.505155, Obj: 0.488548, No Obj: 0.493684, .5R: 0.190476, .75R: 0.000000, count: 21, class_loss = 9709.556641, iou_loss = 62.510742, total_loss = 9772.067383 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.397830, GIOU: 0.329550), Class: 0.523142, Obj: 0.529151, No Obj: 0.504564, .5R: 0.333333, .75R: 0.041667, count: 24, class_loss = 2513.399658, iou_loss = 17.025635, total_loss = 2530.425293 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.458413, GIOU: 0.448662), Class: 0.528318, Obj: 0.475267, No Obj: 0.496144, .5R: 0.500000, .75R: 0.214286, count: 14, class_loss = 615.907593, iou_loss = 1.324829, total_loss = 617.232422 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.203511, GIOU: 0.088152), Class: 0.571786, Obj: 0.614984, No Obj: 0.493900, .5R: 0.000000, .75R: 0.000000, count: 7, class_loss = 9701.244141, iou_loss = 7.185547, total_loss = 9708.429688 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.379489, GIOU: 0.309481), Class: 0.463813, Obj: 0.460192, No Obj: 0.506175, .5R: 0.176471, .75R: 0.000000, count: 17, class_loss = 2525.879150, iou_loss = 10.207520, total_loss = 2536.086670 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.413185, GIOU: 0.343354), Class: 0.477725, Obj: 0.521630, No Obj: 0.497143, .5R: 0.333333, .75R: 0.000000, count: 24, class_loss = 618.620667, iou_loss = 2.274963, total_loss = 620.895630 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.235902, GIOU: 0.170231), Class: 0.533731, Obj: 0.580615, No Obj: 0.493958, .5R: 0.000000, .75R: 0.000000, count: 5, class_loss = 9703.907227, iou_loss = 7.406250, total_loss = 9711.313477 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.384019, GIOU: 0.278774), Class: 0.458359, Obj: 0.508188, No Obj: 0.505172, .5R: 0.125000, .75R: 0.000000, count: 16, class_loss = 2521.044922, iou_loss = 9.494385, total_loss = 2530.539307 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.453716, GIOU: 0.413037), Class: 0.481823, Obj: 0.526693, No Obj: 0.496217, .5R: 0.375000, .75R: 0.000000, count: 8, class_loss = 613.050354, iou_loss = 1.440247, total_loss = 614.490601 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494688, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9739.177734, iou_loss = 0.000000, total_loss = 9739.177734 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.353680, GIOU: 0.224105), Class: 0.558523, Obj: 0.411946, No Obj: 0.505977, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2526.809082, iou_loss = 0.076660, total_loss = 2526.885742 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.441059, GIOU: 0.306029), Class: 0.520586, Obj: 0.465244, No Obj: 0.495875, .5R: 0.250000, .75R: 0.000000, count: 4, class_loss = 612.928528, iou_loss = 0.110168, total_loss = 613.038696 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494937, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9721.110352, iou_loss = 0.000000, total_loss = 9721.110352 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.324500, GIOU: 0.324500), Class: 0.524538, Obj: 0.467388, No Obj: 0.505529, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 2518.706299, iou_loss = 1.758545, total_loss = 2520.464844 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.502427, GIOU: 0.469841), Class: 0.506487, Obj: 0.570413, No Obj: 0.495723, .5R: 0.571429, .75R: 0.000000, count: 7, class_loss = 612.387329, iou_loss = 1.315552, total_loss = 613.702881 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.496100, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9816.364258, iou_loss = 0.000000, total_loss = 9816.364258 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.513060, GIOU: 0.507512), Class: 0.501331, Obj: 0.451795, No Obj: 0.504922, .5R: 0.500000, .75R: 0.000000, count: 4, class_loss = 2518.430664, iou_loss = 2.128906, total_loss = 2520.559570 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.490944, GIOU: 0.467940), Class: 0.501722, Obj: 0.485380, No Obj: 0.495502, .5R: 0.538462, .75R: 0.153846, count: 13, class_loss = 615.688721, iou_loss = 1.547180, total_loss = 617.235901 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.380012, GIOU: 0.340260), Class: 0.509875, Obj: 0.557244, No Obj: 0.495229, .5R: 0.266667, .75R: 0.000000, count: 30, class_loss = 9749.583984, iou_loss = 114.000977, total_loss = 9863.584961 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.333684, GIOU: 0.279693), Class: 0.537264, Obj: 0.439055, No Obj: 0.505427, .5R: 0.200000, .75R: 0.000000, count: 10, class_loss = 2523.791748, iou_loss = 8.799072, total_loss = 2532.590820 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.556861, GIOU: 0.552041), Class: 0.523681, Obj: 0.430879, No Obj: 0.495591, .5R: 0.666667, .75R: 0.333333, count: 3, class_loss = 613.765015, iou_loss = 0.077698, total_loss = 613.842712 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494784, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9729.316406, iou_loss = 0.000000, total_loss = 9729.316406 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.314435, GIOU: 0.247469), Class: 0.460413, Obj: 0.461756, No Obj: 0.506259, .5R: 0.000000, .75R: 0.000000, count: 9, class_loss = 2527.240967, iou_loss = 2.607422, total_loss = 2529.848389 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.325663, GIOU: 0.222342), Class: 0.489587, Obj: 0.474741, No Obj: 0.495887, .5R: 0.136364, .75R: 0.000000, count: 22, class_loss = 620.723206, iou_loss = 0.767700, total_loss = 621.490906 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.493795, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9695.295898, iou_loss = 0.000000, total_loss = 9695.295898 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.504344, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2507.750732, iou_loss = 0.000000, total_loss = 2507.750732 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.494469, GIOU: 0.494469), Class: 0.505453, Obj: 0.397545, No Obj: 0.495561, .5R: 0.666667, .75R: 0.000000, count: 3, class_loss = 611.743591, iou_loss = 0.085022, total_loss = 611.828613 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494902, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9736.839844, iou_loss = 0.000000, total_loss = 9736.839844 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.505327, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2521.004395, iou_loss = 0.000000, total_loss = 2521.004395 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.466371, GIOU: 0.422573), Class: 0.404840, Obj: 0.432268, No Obj: 0.496179, .5R: 0.333333, .75R: 0.000000, count: 3, class_loss = 615.381409, iou_loss = 0.113098, total_loss = 615.494507 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.220952, GIOU: 0.220952), Class: 0.454425, Obj: 0.522251, No Obj: 0.494259, .5R: 0.000000, .75R: 0.000000, count: 4, class_loss = 9701.027344, iou_loss = 4.402344, total_loss = 9705.429688 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.343223, GIOU: 0.243291), Class: 0.416340, Obj: 0.493283, No Obj: 0.504029, .5R: 0.200000, .75R: 0.000000, count: 20, class_loss = 2512.367188, iou_loss = 9.534912, total_loss = 2521.902100 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.337673, GIOU: 0.134180), Class: 0.529619, Obj: 0.519264, No Obj: 0.495715, .5R: 0.142857, .75R: 0.000000, count: 7, class_loss = 613.197266, iou_loss = 0.351807, total_loss = 613.549072 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494781, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9728.414062, iou_loss = 0.000000, total_loss = 9728.414062 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.331892, GIOU: 0.270272), Class: 0.429707, Obj: 0.531512, No Obj: 0.505501, .5R: 0.166667, .75R: 0.000000, count: 12, class_loss = 2526.310791, iou_loss = 4.073730, total_loss = 2530.384521 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.349980, GIOU: 0.293205), Class: 0.508103, Obj: 0.496944, No Obj: 0.496795, .5R: 0.142857, .75R: 0.071429, count: 14, class_loss = 619.674072, iou_loss = 1.409363, total_loss = 621.083435 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.191327, GIOU: -0.132445), Class: 0.524448, Obj: 0.527119, No Obj: 0.494995, .5R: 0.000000, .75R: 0.000000, count: 2, class_loss = 9747.631836, iou_loss = 1.434570, total_loss = 9749.066406 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.358403, GIOU: 0.300473), Class: 0.450436, Obj: 0.529290, No Obj: 0.506191, .5R: 0.307692, .75R: 0.038462, count: 26, class_loss = 2531.524414, iou_loss = 12.167969, total_loss = 2543.692383 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.372848, GIOU: 0.259213), Class: 0.511518, Obj: 0.582628, No Obj: 0.495262, .5R: 0.214286, .75R: 0.000000, count: 14, class_loss = 612.876038, iou_loss = 3.359741, total_loss = 616.235779 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.281748, GIOU: 0.212347), Class: 0.442100, Obj: 0.619671, No Obj: 0.493550, .5R: 0.000000, .75R: 0.000000, count: 4, class_loss = 9679.235352, iou_loss = 4.139648, total_loss = 9683.375000 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.274590, GIOU: 0.257286), Class: 0.511140, Obj: 0.589656, No Obj: 0.505082, .5R: 0.200000, .75R: 0.000000, count: 10, class_loss = 2514.946533, iou_loss = 3.204834, total_loss = 2518.151367 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.382264, GIOU: 0.344977), Class: 0.509203, Obj: 0.511261, No Obj: 0.497097, .5R: 0.142857, .75R: 0.000000, count: 7, class_loss = 616.630005, iou_loss = 0.609253, total_loss = 617.239258 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.170297, GIOU: 0.170298), Class: 0.601900, Obj: 0.597573, No Obj: 0.494499, .5R: 0.000000, .75R: 0.000000, count: 5, class_loss = 9729.549805, iou_loss = 3.227539, total_loss = 9732.777344 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.350111, GIOU: 0.268177), Class: 0.444275, Obj: 0.524385, No Obj: 0.505095, .5R: 0.235294, .75R: 0.000000, count: 17, class_loss = 2521.203125, iou_loss = 8.279541, total_loss = 2529.482666 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.405786, GIOU: 0.332578), Class: 0.484995, Obj: 0.515158, No Obj: 0.494997, .5R: 0.285714, .75R: 0.142857, count: 14, class_loss = 614.453674, iou_loss = 1.996277, total_loss = 616.449951 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.227461, GIOU: 0.060429), Class: 0.557221, Obj: 0.545290, No Obj: 0.494168, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 9719.422852, iou_loss = 3.591797, total_loss = 9723.014648 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.276612, GIOU: 0.188634), Class: 0.499802, Obj: 0.471063, No Obj: 0.506399, .5R: 0.125000, .75R: 0.000000, count: 8, class_loss = 2535.060303, iou_loss = 1.701904, total_loss = 2536.762207 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.385655, GIOU: 0.283922), Class: 0.492178, Obj: 0.496064, No Obj: 0.495976, .5R: 0.100000, .75R: 0.000000, count: 10, class_loss = 616.125854, iou_loss = 0.809509, total_loss = 616.935364 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.156836, GIOU: -0.042966), Class: 0.514550, Obj: 0.455197, No Obj: 0.494530, .5R: 0.000000, .75R: 0.000000, count: 9, class_loss = 9718.803711, iou_loss = 3.512695, total_loss = 9722.316406 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.378920, GIOU: 0.245920), Class: 0.493946, Obj: 0.500731, No Obj: 0.506372, .5R: 0.323529, .75R: 0.029412, count: 34, class_loss = 2536.767578, iou_loss = 24.720947, total_loss = 2561.488525 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.347377, GIOU: 0.196532), Class: 0.533640, Obj: 0.506593, No Obj: 0.496426, .5R: 0.076923, .75R: 0.000000, count: 13, class_loss = 614.961121, iou_loss = 2.791992, total_loss = 617.753113 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.323941, GIOU: 0.193609), Class: 0.638149, Obj: 0.413807, No Obj: 0.495741, .5R: 0.138889, .75R: 0.000000, count: 36, class_loss = 9775.004883, iou_loss = 177.582031, total_loss = 9952.586914 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.291543, GIOU: 0.226092), Class: 0.584305, Obj: 0.540459, No Obj: 0.504691, .5R: 0.000000, .75R: 0.000000, count: 6, class_loss = 2514.697510, iou_loss = 2.932861, total_loss = 2517.630371 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.333456, GIOU: 0.266854), Class: 0.491334, Obj: 0.485666, No Obj: 0.496225, .5R: 0.000000, .75R: 0.000000, count: 8, class_loss = 615.022705, iou_loss = 0.359497, total_loss = 615.382202 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.240913, GIOU: 0.040421), Class: 0.491829, Obj: 0.497619, No Obj: 0.494937, .5R: 0.000000, .75R: 0.000000, count: 17, class_loss = 9732.093750, iou_loss = 18.395508, total_loss = 9750.489258 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.322461, GIOU: 0.177785), Class: 0.493844, Obj: 0.547795, No Obj: 0.504453, .5R: 0.111111, .75R: 0.000000, count: 27, class_loss = 2513.040039, iou_loss = 20.289307, total_loss = 2533.329346 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.472346, GIOU: 0.446014), Class: 0.484881, Obj: 0.523412, No Obj: 0.495963, .5R: 0.400000, .75R: 0.000000, count: 5, class_loss = 610.524536, iou_loss = 0.874329, total_loss = 611.398865 
    
     Tensor Cores are disabled until the first 3000 iterations are reached.
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.333117, GIOU: 0.255250), Class: 0.501665, Obj: 0.554237, No Obj: 0.494657, .5R: 0.138889, .75R: 0.000000, count: 36, class_loss = 9745.872070, iou_loss = 124.244141, total_loss = 9870.116211 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.290920, GIOU: 0.218595), Class: 0.564537, Obj: 0.516925, No Obj: 0.505445, .5R: 0.000000, .75R: 0.000000, count: 18, class_loss = 2524.914062, iou_loss = 6.064453, total_loss = 2530.978516 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.477695, GIOU: 0.394995), Class: 0.525425, Obj: 0.495070, No Obj: 0.496478, .5R: 0.428571, .75R: 0.142857, count: 7, class_loss = 616.713867, iou_loss = 0.833435, total_loss = 617.547302 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.132191, GIOU: 0.132191), Class: 0.475311, Obj: 0.600053, No Obj: 0.495611, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 9750.757812, iou_loss = 1.651367, total_loss = 9752.409180 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.347205, GIOU: 0.238160), Class: 0.450344, Obj: 0.498881, No Obj: 0.505220, .5R: 0.181818, .75R: 0.000000, count: 11, class_loss = 2519.208252, iou_loss = 4.612305, total_loss = 2523.820557 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.320895, GIOU: 0.218805), Class: 0.541832, Obj: 0.522303, No Obj: 0.495714, .5R: 0.142857, .75R: 0.000000, count: 7, class_loss = 614.266174, iou_loss = 0.903809, total_loss = 615.169983 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.220116, GIOU: 0.211029), Class: 0.507227, Obj: 0.537346, No Obj: 0.496046, .5R: 0.250000, .75R: 0.000000, count: 4, class_loss = 9801.845703, iou_loss = 15.615234, total_loss = 9817.460938 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.294504, GIOU: 0.183503), Class: 0.425407, Obj: 0.489159, No Obj: 0.504471, .5R: 0.181818, .75R: 0.000000, count: 11, class_loss = 2521.004395, iou_loss = 4.106934, total_loss = 2525.111328 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.331872, GIOU: 0.186792), Class: 0.556278, Obj: 0.507992, No Obj: 0.495685, .5R: 0.181818, .75R: 0.000000, count: 11, class_loss = 614.859314, iou_loss = 0.486755, total_loss = 615.346069 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494468, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9702.747070, iou_loss = 0.000000, total_loss = 9702.747070 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.498779, GIOU: 0.397074), Class: 0.363107, Obj: 0.485609, No Obj: 0.505086, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2515.734863, iou_loss = 0.301025, total_loss = 2516.035889 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.400137, GIOU: 0.246350), Class: 0.468344, Obj: 0.512063, No Obj: 0.494776, .5R: 0.333333, .75R: 0.000000, count: 3, class_loss = 609.339417, iou_loss = 0.082520, total_loss = 609.421936 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.319060, GIOU: 0.279103), Class: 0.596916, Obj: 0.519859, No Obj: 0.495056, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 9757.434570, iou_loss = 7.479492, total_loss = 9764.914062 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.347641, GIOU: 0.254072), Class: 0.474193, Obj: 0.514575, No Obj: 0.504294, .5R: 0.250000, .75R: 0.000000, count: 8, class_loss = 2510.624023, iou_loss = 5.363037, total_loss = 2515.987061 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.301271, GIOU: 0.267224), Class: 0.477870, Obj: 0.533847, No Obj: 0.494504, .5R: 0.166667, .75R: 0.000000, count: 12, class_loss = 612.643799, iou_loss = 0.577942, total_loss = 613.221741 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.173686, GIOU: 0.088884), Class: 0.574905, Obj: 0.476034, No Obj: 0.493833, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 9695.890625, iou_loss = 1.783203, total_loss = 9697.673828 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.313533, GIOU: 0.247736), Class: 0.438681, Obj: 0.523573, No Obj: 0.504867, .5R: 0.230769, .75R: 0.000000, count: 13, class_loss = 2515.992920, iou_loss = 5.736328, total_loss = 2521.729248 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.390391, GIOU: 0.334270), Class: 0.470548, Obj: 0.540637, No Obj: 0.495472, .5R: 0.272727, .75R: 0.000000, count: 11, class_loss = 614.442810, iou_loss = 1.909485, total_loss = 616.352295 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.294805, GIOU: 0.137412), Class: 0.539223, Obj: 0.577162, No Obj: 0.493229, .5R: 0.111111, .75R: 0.000000, count: 9, class_loss = 9677.886719, iou_loss = 31.590820, total_loss = 9709.477539 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.376017, GIOU: 0.291286), Class: 0.486700, Obj: 0.479627, No Obj: 0.504349, .5R: 0.333333, .75R: 0.000000, count: 9, class_loss = 2506.143066, iou_loss = 4.458252, total_loss = 2510.601318 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.374089, GIOU: 0.357616), Class: 0.474281, Obj: 0.496772, No Obj: 0.496734, .5R: 0.357143, .75R: 0.000000, count: 14, class_loss = 615.890991, iou_loss = 0.825439, total_loss = 616.716431 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.245281, GIOU: 0.070721), Class: 0.568358, Obj: 0.502835, No Obj: 0.493536, .5R: 0.089286, .75R: 0.000000, count: 56, class_loss = 9696.260742, iou_loss = 117.580078, total_loss = 9813.840820 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.331169, GIOU: 0.239746), Class: 0.484767, Obj: 0.511484, No Obj: 0.505126, .5R: 0.235294, .75R: 0.000000, count: 34, class_loss = 2522.168213, iou_loss = 32.150391, total_loss = 2554.318604 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.386221, GIOU: 0.306767), Class: 0.510543, Obj: 0.484515, No Obj: 0.495379, .5R: 0.200000, .75R: 0.000000, count: 10, class_loss = 611.895203, iou_loss = 1.574097, total_loss = 613.469299 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.353182, GIOU: 0.309214), Class: 0.577716, Obj: 0.628421, No Obj: 0.494046, .5R: 0.000000, .75R: 0.000000, count: 5, class_loss = 9695.958984, iou_loss = 14.701172, total_loss = 9710.660156 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.280797, GIOU: 0.124818), Class: 0.464024, Obj: 0.481169, No Obj: 0.506011, .5R: 0.100000, .75R: 0.000000, count: 20, class_loss = 2533.054688, iou_loss = 6.487793, total_loss = 2539.542480 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.477030, GIOU: 0.413796), Class: 0.505820, Obj: 0.511133, No Obj: 0.496041, .5R: 0.600000, .75R: 0.000000, count: 15, class_loss = 613.545349, iou_loss = 1.250183, total_loss = 614.795532 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.098379, GIOU: 0.098378), Class: 0.555847, Obj: 0.640343, No Obj: 0.493610, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9695.449219, iou_loss = 0.144531, total_loss = 9695.593750 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.372713, GIOU: 0.308137), Class: 0.498901, Obj: 0.445171, No Obj: 0.504173, .5R: 0.214286, .75R: 0.000000, count: 14, class_loss = 2510.866211, iou_loss = 7.791748, total_loss = 2518.657959 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.417478, GIOU: 0.347969), Class: 0.488497, Obj: 0.480623, No Obj: 0.496272, .5R: 0.294118, .75R: 0.058824, count: 17, class_loss = 616.652771, iou_loss = 1.130798, total_loss = 617.783569 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.307217, GIOU: 0.264363), Class: 0.489722, Obj: 0.506388, No Obj: 0.493370, .5R: 0.166667, .75R: 0.000000, count: 6, class_loss = 9684.363281, iou_loss = 16.154297, total_loss = 9700.517578 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.329593, GIOU: 0.247494), Class: 0.445429, Obj: 0.457101, No Obj: 0.504958, .5R: 0.200000, .75R: 0.000000, count: 10, class_loss = 2518.771729, iou_loss = 9.680420, total_loss = 2528.452148 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.357102, GIOU: 0.276377), Class: 0.491131, Obj: 0.491516, No Obj: 0.496895, .5R: 0.200000, .75R: 0.000000, count: 20, class_loss = 619.466614, iou_loss = 0.869446, total_loss = 620.336060 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494837, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9755.525391, iou_loss = 0.000000, total_loss = 9755.525391 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.506342, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2533.288574, iou_loss = 0.000000, total_loss = 2533.288574 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.237384, GIOU: 0.237384), Class: 0.484701, Obj: 0.437752, No Obj: 0.496327, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 615.689270, iou_loss = 0.026367, total_loss = 615.715637 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.147036, GIOU: 0.147036), Class: 0.484500, Obj: 0.624998, No Obj: 0.495558, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9767.616211, iou_loss = 0.324219, total_loss = 9767.940430 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.265049, GIOU: 0.212918), Class: 0.364780, Obj: 0.524741, No Obj: 0.505110, .5R: 0.000000, .75R: 0.000000, count: 6, class_loss = 2517.957520, iou_loss = 1.525391, total_loss = 2519.482910 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.336380, GIOU: 0.250431), Class: 0.516341, Obj: 0.470223, No Obj: 0.495021, .5R: 0.000000, .75R: 0.000000, count: 7, class_loss = 612.277344, iou_loss = 0.543640, total_loss = 612.820984 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.203920, GIOU: 0.028839), Class: 0.612857, Obj: 0.511484, No Obj: 0.495206, .5R: 0.000000, .75R: 0.000000, count: 6, class_loss = 9741.394531, iou_loss = 7.089844, total_loss = 9748.484375 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.352583, GIOU: 0.209370), Class: 0.533027, Obj: 0.537346, No Obj: 0.504517, .5R: 0.250000, .75R: 0.000000, count: 8, class_loss = 2509.536377, iou_loss = 5.890625, total_loss = 2515.427002 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.404430, GIOU: 0.361525), Class: 0.511771, Obj: 0.537797, No Obj: 0.495295, .5R: 0.400000, .75R: 0.000000, count: 5, class_loss = 612.462341, iou_loss = 0.495300, total_loss = 612.957642 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.172108, GIOU: 0.154190), Class: 0.525164, Obj: 0.480538, No Obj: 0.495247, .5R: 0.000000, .75R: 0.000000, count: 12, class_loss = 9768.769531, iou_loss = 7.767578, total_loss = 9776.537109 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.393611, GIOU: 0.345943), Class: 0.480370, Obj: 0.519895, No Obj: 0.505282, .5R: 0.283951, .75R: 0.024691, count: 81, class_loss = 2536.175293, iou_loss = 52.745605, total_loss = 2588.920898 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.397607, GIOU: 0.339442), Class: 0.496517, Obj: 0.513687, No Obj: 0.494989, .5R: 0.275000, .75R: 0.050000, count: 40, class_loss = 620.007141, iou_loss = 6.976074, total_loss = 626.983215 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.495312, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9772.366211, iou_loss = 0.000000, total_loss = 9772.366211 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.504319, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 2513.239502, iou_loss = 0.000000, total_loss = 2513.239502 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.499686, GIOU: 0.490835), Class: 0.498215, Obj: 0.467283, No Obj: 0.496509, .5R: 0.666667, .75R: 0.333333, count: 3, class_loss = 616.208191, iou_loss = 0.076477, total_loss = 616.284668 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.211409, GIOU: 0.154423), Class: 0.522371, Obj: 0.437717, No Obj: 0.494731, .5R: 0.000000, .75R: 0.000000, count: 7, class_loss = 9744.777344, iou_loss = 6.811523, total_loss = 9751.588867 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.360554, GIOU: 0.258569), Class: 0.510080, Obj: 0.596456, No Obj: 0.504486, .5R: 0.076923, .75R: 0.000000, count: 13, class_loss = 2509.352539, iou_loss = 9.502197, total_loss = 2518.854736 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.385155, GIOU: 0.361470), Class: 0.468731, Obj: 0.461988, No Obj: 0.495853, .5R: 0.272727, .75R: 0.000000, count: 11, class_loss = 614.686096, iou_loss = 0.602478, total_loss = 615.288574 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.386603, GIOU: 0.322571), Class: 0.495166, Obj: 0.497325, No Obj: 0.493346, .5R: 0.161290, .75R: 0.000000, count: 31, class_loss = 9675.115234, iou_loss = 156.862305, total_loss = 9831.977539 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.359293, GIOU: 0.219897), Class: 0.476842, Obj: 0.495638, No Obj: 0.505006, .5R: 0.200000, .75R: 0.000000, count: 10, class_loss = 2515.837158, iou_loss = 4.922363, total_loss = 2520.759521 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.377328, GIOU: 0.299579), Class: 0.485441, Obj: 0.443205, No Obj: 0.497325, .5R: 0.214286, .75R: 0.071429, count: 14, class_loss = 619.625305, iou_loss = 0.441895, total_loss = 620.067200 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.493999, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9706.401367, iou_loss = 0.000000, total_loss = 9706.401367 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.372961, GIOU: 0.242825), Class: 0.478803, Obj: 0.470151, No Obj: 0.504717, .5R: 0.000000, .75R: 0.000000, count: 6, class_loss = 2512.915527, iou_loss = 2.119629, total_loss = 2515.035156 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.312930, GIOU: 0.291918), Class: 0.508228, Obj: 0.476247, No Obj: 0.495947, .5R: 0.071429, .75R: 0.000000, count: 14, class_loss = 616.366943, iou_loss = 0.785339, total_loss = 617.152283 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.188736, GIOU: 0.122286), Class: 0.387290, Obj: 0.567407, No Obj: 0.493476, .5R: 0.000000, .75R: 0.000000, count: 2, class_loss = 9667.074219, iou_loss = 7.390625, total_loss = 9674.464844 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.330124, GIOU: 0.269697), Class: 0.469831, Obj: 0.403251, No Obj: 0.505506, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 2516.715088, iou_loss = 0.632812, total_loss = 2517.347900 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.352859, GIOU: 0.244853), Class: 0.476575, Obj: 0.469130, No Obj: 0.496894, .5R: 0.153846, .75R: 0.000000, count: 13, class_loss = 615.689392, iou_loss = 0.661072, total_loss = 616.350464 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494105, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9707.502930, iou_loss = 0.000000, total_loss = 9707.502930 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.412769, GIOU: 0.383836), Class: 0.448030, Obj: 0.485255, No Obj: 0.505480, .5R: 0.250000, .75R: 0.000000, count: 4, class_loss = 2514.903076, iou_loss = 2.266113, total_loss = 2517.169189 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.363377, GIOU: 0.309524), Class: 0.499480, Obj: 0.474212, No Obj: 0.495697, .5R: 0.250000, .75R: 0.041667, count: 24, class_loss = 615.788330, iou_loss = 1.120728, total_loss = 616.909058 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.217467, GIOU: 0.036011), Class: 0.555387, Obj: 0.477668, No Obj: 0.494468, .5R: 0.000000, .75R: 0.000000, count: 8, class_loss = 9727.041016, iou_loss = 7.031250, total_loss = 9734.072266 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.411286, GIOU: 0.340007), Class: 0.518636, Obj: 0.488002, No Obj: 0.504496, .5R: 0.333333, .75R: 0.000000, count: 18, class_loss = 2512.063232, iou_loss = 14.034424, total_loss = 2526.097656 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.466993, GIOU: 0.404002), Class: 0.547139, Obj: 0.508334, No Obj: 0.495547, .5R: 0.500000, .75R: 0.083333, count: 12, class_loss = 615.674133, iou_loss = 1.196228, total_loss = 616.870361 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494417, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9723.993164, iou_loss = 0.000000, total_loss = 9723.993164 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.229213, GIOU: 0.229213), Class: 0.459371, Obj: 0.565575, No Obj: 0.505266, .5R: 0.000000, .75R: 0.000000, count: 2, class_loss = 2517.181641, iou_loss = 0.316650, total_loss = 2517.498291 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.345756, GIOU: 0.273917), Class: 0.505451, Obj: 0.500829, No Obj: 0.495800, .5R: 0.250000, .75R: 0.000000, count: 4, class_loss = 614.251099, iou_loss = 0.101868, total_loss = 614.352966 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.233088, GIOU: 0.171825), Class: 0.487095, Obj: 0.610041, No Obj: 0.494668, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 9727.833008, iou_loss = 6.034180, total_loss = 9733.867188 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.270342, GIOU: 0.177598), Class: 0.468114, Obj: 0.483089, No Obj: 0.504984, .5R: 0.052632, .75R: 0.000000, count: 19, class_loss = 2522.070557, iou_loss = 8.694580, total_loss = 2530.765137 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.373883, GIOU: 0.347485), Class: 0.527266, Obj: 0.545368, No Obj: 0.495300, .5R: 0.166667, .75R: 0.000000, count: 12, class_loss = 614.754456, iou_loss = 1.720398, total_loss = 616.474854 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.333033, GIOU: 0.229205), Class: 0.597322, Obj: 0.459012, No Obj: 0.496273, .5R: 0.071429, .75R: 0.000000, count: 14, class_loss = 9819.368164, iou_loss = 30.516602, total_loss = 9849.884766 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.431785, GIOU: 0.357384), Class: 0.495047, Obj: 0.510499, No Obj: 0.504282, .5R: 0.285714, .75R: 0.000000, count: 21, class_loss = 2518.322998, iou_loss = 22.443604, total_loss = 2540.766602 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.452263, GIOU: 0.369422), Class: 0.546137, Obj: 0.509981, No Obj: 0.495228, .5R: 0.250000, .75R: 0.000000, count: 4, class_loss = 611.570557, iou_loss = 0.641113, total_loss = 612.211670 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494001, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9673.214844, iou_loss = 0.000000, total_loss = 9673.214844 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.301258, GIOU: 0.289700), Class: 0.527727, Obj: 0.389444, No Obj: 0.505459, .5R: 0.200000, .75R: 0.200000, count: 5, class_loss = 2517.530762, iou_loss = 1.571777, total_loss = 2519.102539 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.419201, GIOU: 0.340236), Class: 0.501935, Obj: 0.495616, No Obj: 0.495676, .5R: 0.200000, .75R: 0.100000, count: 10, class_loss = 612.381165, iou_loss = 0.824402, total_loss = 613.205566 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.494778, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9774.806641, iou_loss = 0.000000, total_loss = 9774.806641 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.315315, GIOU: 0.189082), Class: 0.466870, Obj: 0.461328, No Obj: 0.506442, .5R: 0.000000, .75R: 0.000000, count: 4, class_loss = 2536.335449, iou_loss = 0.651123, total_loss = 2536.986572 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.387090, GIOU: 0.300122), Class: 0.516208, Obj: 0.498847, No Obj: 0.495934, .5R: 0.375000, .75R: 0.000000, count: 8, class_loss = 614.866150, iou_loss = 0.220459, total_loss = 615.086609 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.000000, GIOU: 0.000000), Class: 0.000000, Obj: 0.000000, No Obj: 0.493374, .5R: 0.000000, .75R: 0.000000, count: 1, class_loss = 9679.333008, iou_loss = 0.000000, total_loss = 9679.333008 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.374473, GIOU: 0.326277), Class: 0.438390, Obj: 0.518467, No Obj: 0.506602, .5R: 0.200000, .75R: 0.000000, count: 10, class_loss = 2528.178955, iou_loss = 3.708740, total_loss = 2531.887695 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.403197, GIOU: 0.244832), Class: 0.491359, Obj: 0.474051, No Obj: 0.496014, .5R: 0.285714, .75R: 0.095238, count: 21, class_loss = 619.201050, iou_loss = 1.601318, total_loss = 620.802368 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.246456, GIOU: 0.077928), Class: 0.582312, Obj: 0.542026, No Obj: 0.494314, .5R: 0.000000, .75R: 0.000000, count: 7, class_loss = 9692.765625, iou_loss = 5.811523, total_loss = 9698.577148 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.363155, GIOU: 0.215500), Class: 0.426346, Obj: 0.560835, No Obj: 0.505772, .5R: 0.166667, .75R: 0.000000, count: 12, class_loss = 2524.271484, iou_loss = 6.414307, total_loss = 2530.685791 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.320936, GIOU: 0.179099), Class: 0.500298, Obj: 0.484651, No Obj: 0.496997, .5R: 0.153846, .75R: 0.000000, count: 13, class_loss = 618.105957, iou_loss = 0.557434, total_loss = 618.663391 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.293965, GIOU: 0.272796), Class: 0.565817, Obj: 0.605423, No Obj: 0.493191, .5R: 0.333333, .75R: 0.000000, count: 3, class_loss = 9682.823242, iou_loss = 6.445312, total_loss = 9689.268555 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.363749, GIOU: 0.307294), Class: 0.471375, Obj: 0.469432, No Obj: 0.505261, .5R: 0.277778, .75R: 0.000000, count: 18, class_loss = 2514.567139, iou_loss = 9.413330, total_loss = 2523.980469 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.413112, GIOU: 0.371660), Class: 0.525244, Obj: 0.485744, No Obj: 0.496948, .5R: 0.230769, .75R: 0.115385, count: 26, class_loss = 619.378357, iou_loss = 1.652832, total_loss = 621.031189 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.242383, GIOU: -0.085741), Class: 0.759331, Obj: 0.406933, No Obj: 0.493504, .5R: 0.000000, .75R: 0.000000, count: 3, class_loss = 9674.526367, iou_loss = 13.637695, total_loss = 9688.164062 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.371533, GIOU: 0.268707), Class: 0.475258, Obj: 0.505309, No Obj: 0.505629, .5R: 0.200000, .75R: 0.000000, count: 5, class_loss = 2514.907471, iou_loss = 1.601807, total_loss = 2516.509277 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.368017, GIOU: 0.294676), Class: 0.506329, Obj: 0.472008, No Obj: 0.496349, .5R: 0.210526, .75R: 0.105263, count: 19, class_loss = 615.307678, iou_loss = 1.017029, total_loss = 616.324707 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 139 Avg (IOU: 0.316925, GIOU: 0.104135), Class: 0.489753, Obj: 0.474642, No Obj: 0.493801, .5R: 0.250000, .75R: 0.000000, count: 4, class_loss = 9655.998047, iou_loss = 9.705078, total_loss = 9665.703125 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 150 Avg (IOU: 0.386729, GIOU: 0.238052), Class: 0.423498, Obj: 0.477232, No Obj: 0.505107, .5R: 0.333333, .75R: 0.000000, count: 6, class_loss = 2508.250488, iou_loss = 5.477051, total_loss = 2513.727539 
    v3 (iou loss, Normalizer: (iou: 0.07, cls: 1.00) Region 161 Avg (IOU: 0.413887, GIOU: 0.388880), Class: 0.518713, Obj: 0.467174, No Obj: 0.494318, .5R: 0.333333, .75R: 0.083333, count: 12, class_loss = 606.953369, iou_loss = 0.440247, total_loss = 607.393616 
    
     Tensor Cores are disabled until the first 3000 iterations are reached.
    


```
!
```
