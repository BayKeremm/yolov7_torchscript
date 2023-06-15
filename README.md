# yolov7_torchscript
YOLOv7 driver code for torchscript \
for speed:
- change torch::tensor to vector of STL, this makes the post processing extremely fast
as done here: https://github.com/BayKeremm/thesis-code/blob/main/image_processing_package/src/Yolo.cpp
