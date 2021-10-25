import torch
import cv2
import numpy as np
import os
from Flownet2Controller_wzg import Flownet2Controller
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# GPU 设置
gpu_index = 0
device = torch.device("cuda:{}".format(gpu_index))

"""
test-1
输入图像路径直接端到端生成光流图
适合单步操作
"""
flownet_path = "./flownet2_pre_train/FlowNet2_checkpoint.pth.tar"
frame1_path = "./datasets/frame_0001.png"
frame2_path = "./datasets/frame_0002.png"
flow_height,flow_width = 384,512

flownet2_controller = Flownet2Controller()
flow_tensor,flow_image = flownet2_controller.flownet2_predict(
                                            flownet_path,
                                            frame1_path,frame2_path,
                                            flow_height,flow_width,
                                            device)

cv2.imwrite("./datasets/flow_test1.jpg",flow_image)

"""
test-2
输入模型及图像张量，生成光流图，适合多次执行
"""
flownet2_controller = Flownet2Controller()
flownet_model = flownet2_controller.load_flownet("./flownet2_pre_train/FlowNet2_checkpoint.pth.tar")
im1,im2 = flownet2_controller.load_frames("./datasets/frame_0001.png","./datasets/frame_0002.png")

# 输入模型及图像张量，生成光流图
flow_tensor = flownet2_controller.gen_flow(flownet_model,im1,im2,384,512,device)

# 光流图转为图像格式
flow_image = flownet2_controller.convert_flow_to_image(flow_tensor)
cv2.imwrite("./datasets/flow_test2.jpg",flow_image)
    
print(im1.shape)
print(flow_tensor.shape)
print(flow_image.shape)