import torch
from torch.autograd import Variable
import argparse
import numpy as np
from os.path import *
import os
import cv2
from torchvision.transforms import Resize 

from src import models, losses, datasets
from src.utils import tools

parser = argparse.ArgumentParser()

parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--total_epochs', type=int, default=10000)
parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
parser.add_argument('--train_n_batches', type=int, default=-1,
                    help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                    help="Spatial dimension to crop training samples for training")
parser.add_argument('--gradient_clip', type=float, default=None)
parser.add_argument('--schedule_lr_frequency', type=int, default=0, help='in number of iterations (0 for no schedule)')
parser.add_argument('--schedule_lr_fraction', type=float, default=10)
parser.add_argument("--rgb_max", type=float, default=255.)

parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
parser.add_argument('--no_cuda', action='store_true')

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
parser.add_argument('--validation_n_batches', type=int, default=-1)
parser.add_argument('--render_validation', action='store_true',
                    help='run inference (save flows to file) and every validation_frequency epoch')

parser.add_argument('--inference', action='store_true')
parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
                    help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
parser.add_argument('--inference_batch_size', type=int, default=1)
parser.add_argument('--inference_n_batches', type=int, default=-1)
parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

parser.add_argument('--skip_training', action='store_true')
parser.add_argument('--skip_validation', action='store_true')

parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--fp16_scale', type=float, default=1024.,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')

tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

tools.add_arguments_for_module(parser, torch.optim, argument_for_class='optimizer', default='Adam',
                               skip_params=['params'])

tools.add_arguments_for_module(parser, datasets, argument_for_class='training_dataset', default='MpiSintelFinal',
                               skip_params=['is_cropped'],
                               parameter_defaults={'root': './MPI-Sintel/flow/training'})

tools.add_arguments_for_module(parser, datasets, argument_for_class='validation_dataset', default='MpiSintelClean',
                               skip_params=['is_cropped'],
                               parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                   'replicates': 1})

tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean',
                               skip_params=['is_cropped'],
                               parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                   'replicates': 1})

args = parser.parse_args()
if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()
parser.add_argument('--IGNORE',  action='store_true')
defaults = vars(parser.parse_args(['--IGNORE']))

args.model_class = tools.module_to_dict(models)[args.model]
args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
args.loss_class = tools.module_to_dict(losses)[args.loss]

args.training_dataset_class = tools.module_to_dict(datasets)[args.training_dataset]
args.validation_dataset_class = tools.module_to_dict(datasets)[args.validation_dataset]
args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.log_file = join(args.save, 'args.txt')

args.grads = {}

if args.inference:
    args.skip_validation = True
    args.skip_training = True
    args.total_epochs = 1
    args.inference_dir = "{}/inference".format(args.save)


args.effective_batch_size = args.batch_size * args.number_gpus
args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
args.effective_number_workers = args.number_workers * args.number_gpus
gpuargs = {'num_workers': args.effective_number_workers,
           'pin_memory': True,
           'drop_last' : True} if args.cuda else {}
inf_gpuargs = gpuargs.copy()
inf_gpuargs['num_workers'] = args.number_workers

class Flownet2Controller():
    def __init__(self):
        # self.flownet_model = self.load_flownet(model_path)
        pass
    def load_flownet(self,flownet_path):
        flownet_model = models.FlowNet2(args)
        checkpoint = torch.load(flownet_path)
        flownet_model.load_state_dict(checkpoint['state_dict'])
        flownet_model.eval()
        return flownet_model

    def gen_flow(self,flownet_model,frame1,frame2,flow_height,flow_width,device):
        Torch_Resize = Resize([flow_height,flow_width])
        frame1 = Torch_Resize(frame1).unsqueeze(2)
        frame2 = Torch_Resize(frame2).unsqueeze(2)

        flownet_model = flownet_model.to(device)
        flow_input = torch.cat([frame1,frame2],axis=2).to(device)

        with torch.no_grad():
            flow = flownet_model(flow_input)
        return flow # shape = [B,C,H,W],C=2

    def convert_flow_to_image(self,flow):
        flow = flow.data.cpu().numpy()[0].transpose(1, 2, 0)
        image_shape = flow.shape[0:2] + (3,)

        hsv = np.zeros(shape=image_shape, dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        normalized_mag = np.asarray(np.clip(mag*40, 0, 255), dtype=np.uint8)
        hsv[..., 2] = normalized_mag
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb = np.asarray(rgb, np.uint8)
        return rgb

    def load_frames(self,frame1_path,frame2_path):
        """
        备注：需要输入图像的scale是[0，255],不能是[0,1]
        """
        im1 = cv2.imread(frame1_path).transpose([2,0,1]) # shapg=[C,H,W]
        im2 = cv2.imread(frame2_path).transpose([2,0,1])
        assert im1.shape == im2.shape

        im1 = torch.from_numpy(im1.astype(np.float32)).unsqueeze(0)# shapg=[B,C,H,W]
        im2 = torch.from_numpy(im2.astype(np.float32)).unsqueeze(0)

        return im1,im2

    def flownet2_predict(self,flownet_path,frame1_path,frame2_path,flow_height,flow_width,device):
        flownet_model = self.load_flownet(flownet_path)
        im1,im2 = self.load_frames(frame1_path,frame2_path)
        flow_tensor = self.gen_flow(flownet_model,im1,im2,flow_height,flow_width,device)
        flow_image = self.convert_flow_to_image(flow_tensor)
        return flow_tensor,flow_image


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # flownet_model = load_flownet("./flownet2_pre_train/FlowNet2_checkpoint.pth.tar")
    # im1,im2 = load_frames("./datasets/frame_0001.png","./datasets/frame_0002.png")
    # flow_tensor = gen_flow(flownet_model,im1,im2,256,256)
    # flow_image = convert_flow_to_image(flow_tensor)
    # cv2.imwrite("./datasets/flow.jpg",flow_image)
        
    # print(im1.shape)
    # print(flow_tensor.shape)
    # print(flow_image.shape)

    gpu_index = 0
    device = torch.device("cuda:{}".format(gpu_index))
    flownet_path = "./flownet2_pre_train/FlowNet2_checkpoint.pth.tar"
    frame1_path = "./datasets/frame_0001.png"
    frame2_path = "./datasets/frame_0002.png"
    flow_height,flow_width = 384,512

    flownet_control = Flownet2Controller()
    flow_tensor,flow_image = flownet_control.flownet2_predict(
                                              flownet_path,
                                              frame1_path,frame2_path,
                                              flow_height,flow_width,device)
    cv2.imwrite("./datasets/flow_3.jpg",flow_image)