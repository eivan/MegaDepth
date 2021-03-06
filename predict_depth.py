import torch
from pathlib import Path
import sys
import math
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions

from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize

def predict_depth(img_path_in):
    
    opt = TrainOptions()
    opt.parser.add_argument('-f', type=str, default='dummy', help='dummy') # needed coz' jupyter
    opt = opt.parse()
    model = create_model(opt)
    
    model.switch_to_eval()

    img = np.float32(io.imread(img_path_in))/255.0
    height, width, depth = img.shape
    
    input_width  = 512# * np.clip(np.floor(width / 512),  1, 2);
    input_height = 384# * np.clip(np.floor(height / 384), 1, 2);
    
    img = resize(img, (input_height, input_width), order = 1)
    input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
    input_img = input_img.unsqueeze(0)

    if torch.cuda.is_available():
      input_images = Variable(input_img.cuda())
    else:
      input_images = Variable(input_img.cpu())
    pred_log_depth = model.netG.forward(input_images) 
    pred_log_depth = torch.squeeze(pred_log_depth)

    pred_depth = torch.exp(pred_log_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    pred_inv_depth = 1/pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

    pred_inv_depth = resize(pred_inv_depth, (height, width), order = 1)
    
    return pred_inv_depth

