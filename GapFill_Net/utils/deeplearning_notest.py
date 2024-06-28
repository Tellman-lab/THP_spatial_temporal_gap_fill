# -*- coding: utf-8 -*-

import logging
import os
import random
import time
from glob import glob

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from albumentations.augmentations import functional as F
from PIL import Image
from pytorch_toolbelt import losses as L
from skimage import io
from torch.autograd import Variable
# from torch.cuda.amp import GradScaler, autocast  # need pytorch>1.6
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
# from torchvision.transforms import functional
from tqdm import tqdm

from utils.utils import AverageMeter, inial_logger, second2time

from .losses import (DiceLoss, FocalLoss, SoftBCEWithLogitsLoss,
                     SoftCrossEntropyLoss, WeightedDiceLoss)
from .metric import IOUMetric, runningScore

# Image.MAX_IMAGE_PIXELS = 1000000000000000
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device='cuda'
def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N).to(device)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)

def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
        
    weights = torch.zeros(target.size(1))

    total_pixels = target.size(0) * target.size(2)

    weights = torch.sum(target,(0,2)) / total_pixels

    weights = 1/(weights**2 + 1e-8)     
        
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
        
    intersection *= weights
    cardinality *= weights
    intersection = torch.sum(intersection)
    cardinality = torch.sum(cardinality)

    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score
def test(val_path,result_path,model):
    # device = torch.device("cuda")
    # setup_seed(1334)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    label_path = os.path.join(val_path,"label")
    rgb_path = os.path.join(val_path,"guangxue")
    dem_path = os.path.join(val_path,"dem")
    slope_path = os.path.join(val_path,"slope")
    aspect_path = os.path.join(val_path,"aspect")
    hig_path = os.path.join(val_path,"hig_dif")
    

    rgb_list = sorted(glob(os.path.join(rgb_path,"*.tif")))
    gt_list = sorted(glob(os.path.join(label_path,"*.tif")))
    dem_list = sorted(glob(os.path.join(dem_path,"*.tif")))
    slope_list = sorted(glob(os.path.join(slope_path,"*.tif")))
    aspect_list = sorted(glob(os.path.join(aspect_path,"*.tif")))
    hig_list = sorted(glob(os.path.join(hig_path,"*.tif")))
   
    # model = DeepLabV3PlusDecoder(7,2).cuda()
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['state_dict'])
    # model = torch.load(model_path)
    # model.eval()
    # model = model.to(device)
    # tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    for idx in range(len(rgb_list)):
        im_dem= io.imread(dem_list[idx])
        im_rgb= io.imread(rgb_list[idx])
        im_slope= io.imread(slope_list[idx])
        im_aspect= io.imread(aspect_list[idx])
        im_hig= io.imread(hig_list[idx])

        
        image = np.concatenate((im_rgb,im_dem[:,:,np.newaxis],im_slope[:,:,np.newaxis],im_aspect[:,:,np.newaxis],im_hig[:,:,np.newaxis]),axis = 2)
       
        
        # image=image.transpose(1, 2, 0)
        image=image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        im = torch.unsqueeze(image,0)
        
        

        im = im.to(device)
        with torch.no_grad():
            # output = tta_model(im)
            output = model(im)

        pred=output.cpu().data.numpy()
        pred= np.argmax(pred,axis=1)

        pred_map = pred.astype("uint8")
        pred_map[pred_map==1]=255
        pred_map = pred_map[0]
        _,new_label_file = os.path.split(gt_list[idx])
        save_path = os.path.join(result_path,new_label_file)
        io.imsave(save_path,pred_map,check_contrast=False)
        # print(save_path)
        

def train_net(param, model, train_data,valid_data,plot=False,device='cuda'):

    model_name      = param['model_name']
    epochs          = param['epochs']
    batch_size      = param['batch_size']
    lr              = param['lr']
    gamma           = param['gamma']
    step_size       = param['step_size']
    momentum        = param['momentum']
    weight_decay    = param['weight_decay']

    disp_inter      = param['disp_inter']
    save_inter      = param['save_inter']
    min_inter       = param['min_inter']
    iter_inter      = param['iter_inter']

    save_log_dir    = param['save_log_dir']
    save_ckpt_dir   = param['save_ckpt_dir']
    load_ckpt_dir   = param['load_ckpt_dir']

    #
    # scaler = GradScaler() 
    # for p in model.parameters():
    #     p.data=p.data.to(torch.float16)

    setup_seed(1334)

    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()
    c, y, x = train_data.__getitem__(0)['image'].shape
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = optim.AdamW(model.parameters(), lr=lr ,weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)
    # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    weights = np.array([0.01,0.99])
    class_wt = torch.from_numpy(weights).to(device,dtype=torch.float32)
    DiceLoss_fn=WeightedDiceLoss(mode='multiclass',class_weights = class_wt)
    BCELoss_fn = nn.CrossEntropyLoss(class_wt)
    SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
    criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                              first_weight=0.5, second_weight=0.5).cuda()
    # criterion = L.JointLoss(first=DiceLoss_fn, second=BCELoss_fn,
    #                           first_weight=0.5, second_weight=0.5).cuda()
    # criterion = SoftCrossEntropy_fn
    # criterion = BCELoss_fn
    logger = inial_logger(os.path.join(save_log_dir, time.strftime("%m-%d-%H-%M-%S", time.localtime()) +'_'+model_name+ '.log'))


    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    valid_loader_size = valid_loader.__len__()
    best_iou = 0
    best_epoch=0
    best_mode = copy.deepcopy(model)
    epoch_start = 0
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        epoch_start = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    logger.info('Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(epochs, x, y, train_data_size, valid_data_size))
    #
    for epoch in tqdm(range(epoch_start, epochs)):
        epoch_start = time.time()

        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target = batch_samples['image'], batch_samples['label']
            data, target = Variable(data.to(device,dtype=torch.float)), Variable(target.to(device,dtype=torch.long))
            # target = get_one_hot(target,2)
            # target = F.one_hot(target,2)
            # target = torch.transpose(target,1,3) 
            # with autocast(): #need pytorch>1.6
            # rgb = data[:,0:3,:,:]
            # dem = data[:,3,:,:]
            # slope = data[:,4,:,:]
            # aspect = data[:,5,:,:]
            # hig = data[:,6,:,:]

            pred = model(data)

            loss = criterion(pred, target)
            # loss = soft_dice_score(pred,target)
            optimizer.zero_grad()
            loss.backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            
            optimizer.step()
            # scheduler.step(epoch + batch_idx / train_loader_size) 
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}s'.format(
                    epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                    optimizer.param_groups[-1]['lr'],
                    train_iter_loss.avg,spend_time / (batch_idx+1) * train_loader_size // 1 - spend_time // 1))
                train_iter_loss.reset()


        # model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        # iou=IOUMetric(2)
        # metrics = SegmentationMetrics()
        # running_metrics_val = runningScore(2)
        # with torch.no_grad():
        #     for batch_idx, batch_samples in enumerate(valid_loader):
        #         data, target = batch_samples['image'], batch_samples['label']
        #         data, target = Variable(data.to(device,dtype=torch.float)), Variable(target.to(device,dtype=torch.long))
        #         # target = get_one_hot(target,2)
        #         # rgb = data[:,0:3,:,:]
        #         # dem = data[:,3,:,:]
        #         # slope = data[:,4,:,:]
        #         # aspect = data[:,5,:,:]
        #         # hig = data[:,6,:,:]
        #         pred = model(data)
        #         # pred = model(data[:,0:3,:,:],data[:,3,:,:])
        #         # pred = model(data[:,:,:,12],data[:,:,:,0:12])
        #         loss = criterion(pred, target)
        #         # loss = soft_dice_score(pred,target)

        
        #         #
        #         pred=pred.cpu().data.numpy()
        #         pred= np.argmax(pred,axis=1)
        #         # pred_ = F.sigmoid(pred)
        #         # pred_ = np.where(pred_>0.5,np.zeros_like(pred_),np.zeros_like(pred_))
        #         iou.add_batch(pred,target.cpu().data.numpy())
        #         # running_metrics_val.update(target.cpu().data.numpy(), pred)
        #         image_loss = loss.item()
        #         valid_epoch_loss.update(image_loss)
        #         valid_iter_loss.update(image_loss)
        #         # if batch_idx % iter_inter == 0:
        #         #     logger.info('[val] epoch:{} iter:{}/{} {:.2f}% loss:{:.6f}'.format(
        #         #         epoch, batch_idx, valid_loader_size, batch_idx / valid_loader_size * 100, valid_iter_loss.avg))
        #     val_loss=valid_iter_loss.avg
        #     # metrics()
            
        #     acc, acc_cls,recall_cls,mean_f1,mean_iu = iou.evaluate()
        #     logger.info('[val] epoch:{} acc:{:.4f} acc_cls:{:.4f} recall_cls:{:.4f} mean_f1:{:.4f} mean_IoU:{:.4f}'.format(epoch,acc,acc_cls,recall_cls,mean_f1,mean_iu))

            # score, class_iou = running_metrics_val.get_scores()
            # for k, v in score.items():
            #     # print(k, v)
            #     logger.info("{}: {}".format(k, v)) 


        train_loss_total_epochs.append(train_epoch_loss.avg)
        # valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])

        # image_path = r"D:\Users\u_deeplabv3\landslide_images\256\test"
        # save_path = r"D:\Users\u_deeplabv3\results"

        if epoch % save_inter == 0 and epoch > min_inter:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)  
            # torch.save(model, filename) 

        # if mean_iu > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            
        #     # print("Inferencing")
        #     # test(image_path,save_path,model)
        #     state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        #     filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
        #     torch.save(state, filename)
        #     # filename_ = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        #     # torch.save(model, filename) 
        #     # torch.save(model, filename_) 
        #     best_iou = mean_iu
        #     best_mode = copy.deepcopy(model)
        #     logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
            
        scheduler.step()
        
        # model_path = r"D:\Users\u_deeplabv3\best_model.pth"
        # model_path = r"D:\Users\u_deeplabv3\checkpoints\UDeepLabV3Plus_\ckpt\checkpoint-best.pth"
        

    # print(best_iou)
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='train loss')
        ax.plot(x, smooth(valid_loss_total_epochs, 0.6), label='val loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('CrossEntropy', fontsize=15)
        ax.set_title('train curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title('lr curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.savefig("learning rate.png", dpi = 300)
        # plt.show()
            
    return model

