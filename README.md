# Flownet2-Pytorch-EasyToUse-Wrapper-wzg16
NVIDIA提供的flownet2-Pytorch中存在一些问题，难以按照他们提供的安装步骤顺利安装。我根据网上的资料及自己的实践整理了一份flownet2-Pytorch的安装教程，见：[flownet2-pytorch安装](https://github.com/wzg16/FlowNet2-PyTorch-EasyToUse-Wrapper-wzg16/blob/master/src/flownet-%E5%AE%89%E8%A3%85).  
由于我只需要使用flownet2-Pytorch，而不需要对其进行改进或重新训练，因此我把NVIDIA版本的flownet2-Pytorch进行了封装,使得可以直接使用flownet2的预训练模型。本文主要介绍我封装后的flownet2的安装与使用教程。  
  

# 安装步骤
## step1:下载代码
```bash
git clone https://github.com/FlowNet2-PyTorch-EasyToUse-Wrapper-wzg16.git
cd FlowNet2-PyTorch-EasyToUse-Wrapper-wzg16
```

## step2: 创建环境
```
conda create -n flownet2_pytorch_env python=3.7  
conda activae flownet2_pytorch_env
```

## step3: 安装依赖包
```
conda install pytorch==1.9.0 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch  
pip install numpy  
pip install tensorboardX  
pip install setproctitle  
pip install colorama  
pip install tqdm  
pip install scipy  
pip install matplotlib  
pip install pytz  
pip install opencv-python  
```

## step4: 安装flownet2_pytorch
```
cd src
./install.sh
```

## step5: 下载checkpoints,放入folder ./flownet2_pre_train 
**FlowNet2S**   
**FlowNet2C**  
**FlowNet2CS**  
**FlowNet2CSS**  
**FlowNet2SD**  
**FlowNet2**  
链接：https://pan.baidu.com/s/19g9ufG5zPARxMFireouXXQ  <br>
提取码：w97r <br>

## step6:测试
```
cd .. 
python flownet2_test_wzg.py
```
# 运行示例
![test](https://github.com/wzg16/FlowNet2-PyTorch-EasyToUse-Wrapper-wzg16/blob/master/datasets/flow_test1.jpg)

# Reference
If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper:

@InProceedings{IMKDB17,<br>
  author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",<br>
  title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",<br>
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",<br>
  month        = "Jul",<br>
  year         = "2017",<br>
  url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"<br>
}<br>
@misc{flownet2-pytorch,<br>
  author = {Fitsum Reda and Robert Pottorff and Jon Barker and Bryan Catanzaro},<br>
  title = {flownet2-pytorch: Pytorch implementation of FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks},<br>
  year = {2017},<br>
  publisher = {GitHub},<br>
  journal = {GitHub repository},<br>
  howpublished = {\url{https://github.com/NVIDIA/flownet2-pytorch}}<br>
}<br>
