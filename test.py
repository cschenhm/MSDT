import os
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data
from model import MultiscaleNet as mynet
from skimage import img_as_ubyte
from get_parameter_number import get_parameter_number
from tqdm import tqdm
from layers import *

parser = argparse.ArgumentParser(description='Image Deraining')
parser.add_argument('--input_dir', default='/data0/chenxiang/code/CVPR2024/Datasets/Rain200H/val/input/', type=str, help='Directory of validation images')
parser.add_argument('--output_dir', default='./results/Rain200H', type=str, help='Directory of validation images')
parser.add_argument('--weights', default='', type=str, help='Path to weights') 
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--win_size', default=256, type=int, help='window size, [GoPro, HIDE, RealBlur]=256, [DPDD]=512')
args = parser.parse_args()
result_dir = args.output_dir
win = args.win_size
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
model_restoration = mynet()
get_parameter_number(model_restoration)
utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# dataset = args.dataset
rgb_dir_test = args.input_dir
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)


utils.mkdir(result_dir)

with torch.no_grad():
    psnr_list = []
    ssim_list = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        input_    = data_test[0].cuda()
        filenames = data_test[1]
        _, _, Hx, Wx = input_.shape
        filenames = data_test[1]
        input_re, batch_list = window_partitionx(input_, win)
        restored = model_restoration(input_re)
        restored = window_reversex(restored[0], win, Hx, Wx, batch_list)

        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        for batch in range(len(restored)):
            restored_img = restored[batch]
            restored_img = img_as_ubyte(restored[batch])
            utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
