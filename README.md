# Flownet2-Pytorch-EasyToUse-Wrapper-wzg16

# Installation:
## step1:下载代码
git clone https://github.com/FlowNet2-PyTorch-EasyToUse-Wrapper-wzg16.git <br>
cd FlowNet2-PyTorch-EasyToUse-Wrapper-wzg16

## step2: 创建环境
`<conda create -n flownet2_pytorch_env python=3.7>` <br>
conda activae flownet2_pytorch_env

## step3: 安装依赖包
conda install pytorch==1.9.0 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch <br>
pip install numpy <br>
pip install tensorboardX <br>
pip install setproctitle <br>
pip install colorama <br>
pip install tqdm <br>
pip install scipy <br>
pip install matplotlib <br>
pip install pytz <br>
pip install opencv-python <br>

## step4: 安装flownet2_pytorch
cd src <br>
./install.sh <br>

## step5: 下载checkpoints,放入folder ./flownet2_pre_train 
FlowNet2S <br>
FlowNet2C <br>
FlowNet2CS <br>
FlowNet2CSS <br>
FlowNet2SD <br>
FlowNet2 <br>
链接：https://pan.baidu.com/s/19g9ufG5zPARxMFireouXXQ  <br>
提取码：w97r <br>

## step6:测试
cd .. <br>
python flownet2_test_wzg.py <br>


# Reference
If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper:

@InProceedings{IMKDB17,
  author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
  title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month        = "Jul",
  year         = "2017",
  url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
}
@misc{flownet2-pytorch,
  author = {Fitsum Reda and Robert Pottorff and Jon Barker and Bryan Catanzaro},
  title = {flownet2-pytorch: Pytorch implementation of FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVIDIA/flownet2-pytorch}}
}
