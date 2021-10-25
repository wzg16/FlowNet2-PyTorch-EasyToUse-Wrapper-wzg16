Flownet2-Pytorch-EasyToUse-Wrapper-wzg16

Installation:
step1:下载代码
git clone https://github.com/FlowNet2-PyTorch-EasyToUse-Wrapper-wzg16.git
cd FlowNet2-PyTorch-EasyToUse-Wrapper-wzg16

step2: 创建环境
conda create -n flownet2_pytorch_env python=3.7
conda activae flownet2_pytorch_env

step3: 安装依赖包
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

step4: 安装flownet2_pytorch
cd src
./install.sh

step5: 下载checkpoints,放入folder ./flownet2_pre_train 
FlowNet2S
FlowNet2C
FlowNet2CS
FlowNet2CSS
FlowNet2SD
FlowNet2
链接：https://pan.baidu.com/s/19g9ufG5zPARxMFireouXXQ 
提取码：w97r

step6:测试
cd ..
python flownet2_test_wzg.py
