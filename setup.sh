#! /bin/bash

# check if folder for model exist, othrewise download it
if [ ! -d "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8" ]; then
	wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
	tar -xzvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
fi

pip install requirements.txt

git clone https://github.com/tensorflow/models.git
cd models/research

# Install protobuf compiler
sudo dnf install -y protobuf-compiler

# Compile the protobuf files
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
# Install the Object Detection API
pip install .
