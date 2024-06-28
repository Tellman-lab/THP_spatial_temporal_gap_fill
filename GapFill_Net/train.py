
import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
#
# import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image



from fusenet import FuseNet

from utils import train_net




Image.MAX_IMAGE_PIXELS = 1000000000000000

device = torch.device("cuda")


import glob

import numpy as np


def main():
    

    # model = model.cuda()
    model = FuseNet(2,1,False).cuda()

    model_name = "fusenet"
    
    # model = UNet(5,2).cuda()
    # model_name = "UNet_fix"

    save_ckpt_dir = os.path.join('checkpoints', model_name, 'ckpt')
    save_vis_dir = os.path.join('checkpoints', model_name, 'vis')
    save_log_dir = os.path.join('checkpoints', model_name)
    if not os.path.exists(save_ckpt_dir):
        os.makedirs(save_ckpt_dir)
    if not os.path.exists(save_vis_dir):
        os.makedirs(save_vis_dir)
    if not os.path.exists(save_log_dir):
        os.makedirs(save_log_dir)


    param = {}

    param['epochs'] = 50        
    param['batch_size'] = 1
    param['lr'] = 1e-4            
    param['gamma'] = 0.5          
    param['step_size'] = 20        
    param['momentum'] = 0.9       
    param['weight_decay'] = 5e-4    
    param['disp_inter'] = 5       
    param['save_inter'] = 10       
    param['iter_inter'] = 50     
    param['min_inter'] = 500

    param['model_name'] = model_name          
    param['save_log_dir'] = save_log_dir      
    param['save_ckpt_dir'] = save_ckpt_dir    


    param['load_ckpt_dir'] = None

    data_root = r'/home/convlstm_predict/FuseNet_Interp_v1/custom_fl_dataset/time_series_dataset_v1/CMO'
    events = glob.glob(data_root+os.sep+"series/2019_03_15/labels/*.tif")
    events = [x.split("/")[-1] for x in events]
    
    test_event = "0a7e.tif"
    model = train_net(param, model, data_root,events,test_event,plot=False)
    
    torch.save(model, "best_model.pth")
    # print(best_miou)
if __name__ == '__main__':
    print("start training...")
    main()
    print("finish training...")
