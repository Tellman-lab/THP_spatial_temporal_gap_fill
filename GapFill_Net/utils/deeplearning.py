# -*- coding: utf-8 -*-

import copy
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

from skimage import io
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast  # need pytorch>1.6
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
# from torchvision.transforms import functional
from tqdm import tqdm

from utils.utils import AverageMeter, inial_logger, second2time

from .losses import (DiceLoss, FocalLoss, SoftBCEWithLogitsLoss,
                     SoftCrossEntropyLoss, WeightedDiceLoss)
from .metric import IOUMetric, runningScore

from dataloader_flood_conv import load_data
from dataloader_flood_conv import create_loader
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


def train_net(param, model, data_root,events,test_event,plot=False,device='cuda'):

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


    use_amp = False
    setup_seed(1334)
    step = 1
    
    optimizer = optim.AdamW(model.parameters(), lr=lr ,weight_decay=weight_decay)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    weights = np.array([0.01,0.99])
    class_wt = torch.from_numpy(weights).to(device,dtype=torch.float32)

    criterion = nn.NLLLoss(weight=class_wt)

    logger = inial_logger(os.path.join(save_log_dir, time.strftime("%m-%d-%H-%M-%S", time.localtime()) +'_'+model_name+ '.log'))


    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []

    best_iou = 0
    best_epoch=0
    best_mode = copy.deepcopy(model)
    epoch_start = 0
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        epoch_start = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # logger.info('Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(epochs, x, y, train_data_size, valid_data_size))
    #
    scaler = GradScaler()
    #
    for epoch in tqdm(range(epoch_start, epochs)):
        # events = ['0a7e.tif','1ilu.tif','3ofd.tif']
        for event in events:

            if event == "g1ot.tif" or event==test_event:
                continue

            print(event+">>"*50)
            train_set, _ = \
                    load_data(data_root=data_root,
                    event=event,
                    idx_in=[-1,0,1],
                    idx_out=0,
                    step=step, use_augment=False)

            timestep,c,y, x = train_set.__getitem__(0)['image'].shape

            train_loader = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,)



            train_loader_size = train_loader.__len__()


            epoch_start = time.time()

            model.train()
            train_epoch_loss = AverageMeter()
            train_iter_loss = AverageMeter()
            for batch_idx, batch_samples in enumerate(train_loader):
                data, target,aux = batch_samples['image'], batch_samples['label'],batch_samples['aux']
                data, target,aux = Variable(data.to(device,dtype=torch.float)), Variable(target.to(device,dtype=torch.long)),Variable(aux.to(device,dtype=torch.float))
    
                optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast():
                    
                        pred = model(data[0][0].unsqueeze(0),data[0][1].unsqueeze(0),aux)
                        loss = criterion(pred, target[:,0,:,:])

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    # pred = model(data[0][0].unsqueeze(0),data[0][1].unsqueeze(0),aux)
                    pred = model(data[:,0,:,:,:],data[:,1,:,:,:],aux)
                    loss = criterion(pred, target[:,0,:,:])
                    loss.backward()
                    optimizer.step()
                    
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


        model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        iou=IOUMetric(2)

        _, test_set = \
                load_data(data_root=data_root,
                event=test_event,
                idx_in=[-1,0,1],
                idx_out=0,
                step=step, use_augment=False)


        valid_data_size = test_set.__len__()
        timestep,c,y, x = train_set.__getitem__(0)['image'].shape


        valid_loader = create_loader(test_set, # validation_set,
                                        batch_size=batch_size,
                                        shuffle=False, is_training=False,
                                        pin_memory=True, drop_last=True,
                                        )
        # metrics = SegmentationMetrics()
        # running_metrics_val = runningScore(2)
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data, target,aux = batch_samples['image'], batch_samples['label'],batch_samples['aux']
                data, target,aux = Variable(data.to(device,dtype=torch.float)), Variable(target.to(device,dtype=torch.long)),Variable(aux.to(device,dtype=torch.float))
    
                # pred = model(data[0][0].unsqueeze(0),data[0][1].unsqueeze(0),aux)
                pred = model(data[:,0,:,:,:],data[:,1,:,:,:],aux)
                loss = criterion(pred, target[:,0,:,:])
                # loss = criterion(pred, target)



                pred=pred.cpu().data.numpy()
                pred= np.argmax(pred,axis=1)

                iou.add_batch(pred,target.cpu().data.numpy())
                # running_metrics_val.update(target.cpu().data.numpy(), pred)
                image_loss = loss.item()
                valid_epoch_loss.update(image_loss)
                valid_iter_loss.update(image_loss)
                # if batch_idx % iter_inter == 0:
                #     logger.info('[val] epoch:{} iter:{}/{} {:.2f}% loss:{:.6f}'.format(
                #         epoch, batch_idx, valid_loader_size, batch_idx / valid_loader_size * 100, valid_iter_loss.avg))
            val_loss=valid_iter_loss.avg

            
            acc, acc_cls,recall_cls,mean_f1,mean_iu = iou.evaluate()
            logger.info('[val] epoch:{} acc:{:.4f} acc_cls:{:.4f} recall_cls:{:.4f} mean_f1:{:.4f} mean_IoU:{:.4f}'.format(epoch,acc,acc_cls,recall_cls,mean_f1,mean_iu))

            # score, class_iou = running_metrics_val.get_scores()
            # for k, v in score.items():
            #     # print(k, v)
            #     logger.info("{}: {}".format(k, v)) 


        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])



        if epoch % save_inter == 0 and epoch > min_inter:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)  
            # torch.save(model, filename) 

        if mean_iu > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            
            # print("Inferencing")
            # test(image_path,save_path,model)
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            # filename_ = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            # torch.save(model, filename) 
            # torch.save(model, filename_) 
            best_iou = mean_iu
            best_mode = copy.deepcopy(model)
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
            
        scheduler.step()
        

        
    train_loss_ = np.array(train_loss_total_epochs)
    val_loss_ = np.array(valid_loss_total_epochs)
    lr_ = np.array(epoch_lr)
    
    np.savetxt("train_loss_.csv", train_loss_, delimiter=',')
    np.savetxt("val_loss_.csv", val_loss_, delimiter=',')
    np.savetxt("lr_.csv", lr_, delimiter=',')

    train_loss_sm = np.array(smooth(train_loss_total_epochs, 0.6))
    val_loss_sm = np.array(smooth(valid_loss_total_epochs, 0.6))

    np.savetxt("train_loss_sm.csv", train_loss_sm, delimiter=',')
    np.savetxt("val_loss_sm.csv", val_loss_sm, delimiter=',')
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
            
    return best_iou,best_mode, model

