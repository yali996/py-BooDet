# py-HAR
This is the code for BooDet, which is a multi-stage object detection method.

# BooDet: Gradient Boosting Object Detection with Additive Learning Based Prediction Aggregation


# Introduction

BooDet is an object detection method which integrates the gradient Boosting for additive learning. 
This is a Caffe implementation, which is modified from the cascade-rcnn. 

This code is only released for academic use. 

# Setting up py-BooDet

## Step 0: Clone this repo and copy files to the required dictionary.
`git clone https://github.com/yali996/py-BooDet.git`

## Step 1: Compile the master branch of caffe
Please copy the .cpp .cu files to the /caffe/src/caffe/layers/
Please copy the .cpp .cu files to the /caffe/include/caffe/layers/
Then checkout the master branch of Caffe and compile it on your machine. Make sure that Caffe must be built with support for Python layers!

## Step 2: Compile the nms module
`cd lib`

`make`

Models is coming soon for testing.
 
# Training
The traning scrip file is coming soon.
