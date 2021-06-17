# Masked Face Detection System

## About

This is an individual assignment for WIX3001 Soft Computing, where soft computing 
technique can be apply to COVID-19.

The system developed able to perform face mask detection with 99% 
accuracy based on the classification report.

## Objectives

1. To identify whether people are masked or unmasked through images captured in real life.

2. To evaluate the accuracy of each classification.

## Set up

Example of Set Up (Project Directory)

[Google Drive Link](https://drive.google.com/drive/folders/1lYKJ4hqSO5F_z1x4NeOsb7nfJQMD5O_M?usp=sharing)

### Packages

The entire project is written in Python. Several packages are used.

1. keras
2. matplotlib
3. numpy
4. OpenCV
5. os
6. random
7. sklearn
8. tensorflow

### Dataset Directory

The dataset is not included in this github repository due to large file. See the ways below.

#### First way (From Kaggle)

The dataset directory need to be set up in the way that it consists 
of another two directories called Masked and Unmasked directory.
The Masked directory consists of 5883 masked face images, while the Unmasked
directory consists of 5909 unmasked face images.

Noted:
   
If you are downloading the dataset from this [link](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset),
you need to combine the image folder, where three sets (Test, Train and Validation) of
WithMask images and WithoutMask images are combined to become one set. All images from
WithMask directory put into the Masked folder, while all images from WithoutMask
directory put into the Unmasked folder.

#### Second way (From Google Drive)

Download the zip file from this [link](https://drive.google.com/drive/folders/1ZLX8XKxKHAoNOOfuUM60Jbl6qzR_z8Aq?usp=sharing). Then unzip locally.

### File set up before training process

Set up the structure of the project that consists several important folder.
The folders needed are:

1. dataset </br>
    > Masked

    > Unmasked
2. input (Empty folder)
3. model (Empty folder)
4. OpenCV DNN (consists of two file, you can check the folder.)
5. output (Empty folder)
6. plot (Empty folder)

### Online Training (Google Colab)

[Link to Google Colab](https://colab.research.google.com/drive/1esp0jAznYRyntjn5hqKNGNwaU8SvaYR2?usp=sharing)

You need to copy the google colab before training. Please make sure to set up the file path correctly.

### Input/Output

You can put images into the input folder. The output can be viewed inside the output folder.

## Experiment Result

Input (face without mask)

<img src="https://github.com/Ken-NPK/Masked-Face-Detection-System/blob/0d44f4213bb3052c91efc06d3b559210a124c32f/input/NG_PHOON_KEN_unmasked.jpg" alt="Input Without Mask" width=50% height=50%>

Output (face without mask)

<img src="https://github.com/Ken-NPK/Masked-Face-Detection-System/blob/0d44f4213bb3052c91efc06d3b559210a124c32f/output/(Predicted)%20NG_PHOON_KEN_unmasked.jpg" alt="Output Without Mask" width=50% height=50%>

Input (face with mask)

<img src="https://github.com/Ken-NPK/Masked-Face-Detection-System/blob/0d44f4213bb3052c91efc06d3b559210a124c32f/input/NG_PHOON_KEN_masked.png" alt="Input With Mask" width=50% height=50%>

Output (face with mask)

<img src="https://github.com/Ken-NPK/Masked-Face-Detection-System/blob/0d44f4213bb3052c91efc06d3b559210a124c32f/output/(Predicted)%20NG_PHOON_KEN_masked.png" alt="Output With Mask" width=50% height=50%>

## Data Source

For this project, I use the image dataset called "Face Mask Detection ~ 12K Images Dataset"

[Dataset Link](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)

