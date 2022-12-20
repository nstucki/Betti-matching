import logging
import os
import yaml
import sys
import random
import numpy as np
import json
from argparse import ArgumentParser
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
from shutil import copyfile
from glob import glob
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
)
from monai.visualize import plot_2d_or_3d_image

from loss_functions import *
from metrics.ClDice.cldice_loss.pytorch.cldice import soft_dice_cldice
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# from topoloss_pytorch import HuTopoLoss


parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yaml) containing the hyper-parameters for training.')
parser.add_argument('--dataconfig',
                    default=None,
                    help='data config file (.yaml) containing the dataset specific information.')
parser.add_argument('--pretrained', default=None, help='checkpoint of the pretrained model')
parser.add_argument('--resume', default=None, help='checkpoint of the last epoch of the model')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0],
                        help='list of index where skip conn will be made')

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def main(args):
# def main(tempdir, max_epoch=100, use_loss=None, alpha=None, pretrained=None, resume=None, name=None):
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)
    
    with open(args.dataconfig) as f:
        print('\n*** Dataconfig file')
        print(args.dataconfig)
        dataconfig = yaml.load(f, Loader=yaml.FullLoader)
    dataconfig = dict2obj(dataconfig)

    # fixing seed for reproducibility
    random.seed(config.TRAIN.SEED)
    np.random.seed(config.TRAIN.SEED)
    torch.random.manual_seed(config.TRAIN.SEED)
    
    if args.resume and args.pretrained:
        raise Exception('Do not use pretrained and resume at the same time.')
    

    # monai.config.print_config()
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # create a temporary directory and 40 random image, mask pairs
    data_path = dataconfig.DATA.DATA_PATH

    images = sorted(glob(os.path.join(data_path+'images', "*"+dataconfig.DATA.FORMAT)))
    segs = sorted(glob(os.path.join(data_path+'labels', "*"+dataconfig.DATA.FORMAT)))
    
    # train and validation files
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:dataconfig.DATA.TRAIN_SAMPLES], segs[:dataconfig.DATA.TRAIN_SAMPLES])]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-dataconfig.DATA.VAL_SAMPLES:], segs[-dataconfig.DATA.VAL_SAMPLES:])]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AddChanneld(keys=["img", "seg"]), # need to check for new dataset
            ScaleIntensityd(keys=["img", "seg"]), # doing normalisation here :)
            RandCropByPosNegLabeld(
                keys=["img", "seg"],
                label_key="seg",
                spatial_size=dataconfig.DATA.IMG_SIZE,
                pos=1,
                neg=1,
                num_samples=dataconfig.DATA.NUM_PATCH,
            ),
            #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AddChanneld(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]), # doing normalisation here :)
            EnsureTyped(keys=["img", "seg"]),
        ]
    )

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.TRAIN.NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=config.TRAIN.BATCH_SIZE,
                            num_workers=config.TRAIN.NUM_WORKERS,
                            collate_fn=list_data_collate)
    
    dice_metric = DiceMetric(include_background=True,
                             reduction="mean",
                             get_not_nans=False)
    
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)
    model = monai.networks.nets.UNet(
        spatial_dims=dataconfig.DATA.DIM,
        in_channels=dataconfig.DATA.IN_CHANNELS,
        out_channels=dataconfig.DATA.OUT_CHANNELS,
        channels=config.MODEL.CHANNELS,
        strides=config.MODEL.STRIDES,
        num_res_units=config.MODEL.NUM_RES_UNITS,
    ).to(device)
    
    # Loss function choice
    if config.LOSS.USE_LOSS == 'Dice':
        if args.pretrained:
            exp_name = config.LOSS.USE_LOSS
        else:
            exp_name = config.LOSS.USE_LOSS + '_scratch'
        loss_function = monai.losses.DiceLoss(sigmoid=True)
    if config.LOSS.USE_LOSS  == 'BettiMatching':
        if args.pretrained:
            exp_name = config.LOSS.USE_LOSS+'_'+config.LOSS.FILTRATION+'_relative_'+str(config.LOSS.RELATIVE)+'_alpha_'+str(config.LOSS.ALPHA)
        else:
            exp_name = config.LOSS.USE_LOSS+'_'+config.LOSS.FILTRATION+'_relative_'+str(config.LOSS.RELATIVE)+'_alpha_'+str(config.LOSS.ALPHA)+'_scratch'
        loss_function = BettiMatchingLoss(relative=config.LOSS.RELATIVE,
                                          filtration=config.LOSS.FILTRATION)
    if config.LOSS.USE_LOSS  == 'DiceBettiMatching':
        if args.pretrained:
            exp_name = config.LOSS.USE_LOSS+'_'+config.LOSS.FILTRATION+'_relative_'+str(config.LOSS.RELATIVE)+'_alpha_'+str(config.LOSS.ALPHA)
        else:
            exp_name = config.LOSS.USE_LOSS+'_'+config.LOSS.FILTRATION+'_relative_'+str(config.LOSS.RELATIVE)+'_alpha_'+str(config.LOSS.ALPHA)+'_scratch'
        loss_function = DiceBettiMatchingLoss(alpha=config.LOSS.ALPHA,
                                              relative=config.LOSS.RELATIVE,
                                              filtration=config.LOSS.FILTRATION)
    if config.LOSS.USE_LOSS  == 'HuTopo':
        if args.pretrained:
            exp_name = config.LOSS.USE_LOSS+'_dimensions_'+str(config.LOSS.DIMENSIONS)+'_alpha_'+str(config.LOSS.ALPHA)
        else:
            exp_name = config.LOSS.USE_LOSS+'_dimensions_'+str(config.LOSS.DIMENSIONS)+'_alpha_'+str(config.LOSS.ALPHA)+'_scratch'
        loss_function = DiceWassersteinLoss(alpha=config.LOSS.ALPHA,
                                       dimensions=config.LOSS.DIMENSIONS)
    if config.LOSS.USE_LOSS == 'ClDice':
        if args.pretrained:
            exp_name = config.LOSS.USE_LOSS+'_alpha_'+str(config.LOSS.ALPHA)
        else:
            exp_name = config.LOSS.USE_LOSS+'_alpha_'+str(config.LOSS.ALPHA)+'_scratch'
        loss_function = soft_dice_cldice(alpha=config.LOSS.ALPHA)
    if config.LOSS.USE_LOSS  == 'ComposedHuTopo':
        if args.pretrained:
            exp_name = config.LOSS.USE_LOSS+'_dimensions_'+str(config.LOSS.DIMENSIONS)+'_alpha_'+str(config.LOSS.ALPHA)
        else:
            exp_name = config.LOSS.USE_LOSS+'_dimensions_'+str(config.LOSS.DIMENSIONS)+'_alpha_'+str(config.LOSS.ALPHA)+'_scratch'
        loss_function = DiceComposedWassersteinLoss(alpha=config.LOSS.ALPHA,
                                       dimensions=config.LOSS.DIMENSIONS)

    # Copy config files and verify if files exist
    exp_path = './models/'+dataconfig.DATA.DATASET+'/'+exp_name
    if os.path.exists(exp_path) and args.resume == None:
        raise Exception('ERROR: Experiment folder exist, please delete folder or check config file')
    else:
        try:
            os.makedirs(exp_path)
            copyfile(args.config, os.path.join(exp_path, "config.yaml"))
        except:
            pass
        
    optimizer = torch.optim.Adam(model.parameters(), config.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.1)   #always check that the step size is high enough
    
    # Resume training
    last_epoch = 0
    if args.resume:
        dic = torch.load(args.resume)
        model.load_state_dict(dic['model'])
        optimizer.load_state_dict(dic['optimizer'])
        scheduler.load_state_dict(dic['scheduler'])
        last_epoch = int(scheduler.last_epoch/len(train_loader))
        
    # Start from pretrained model
    if args.pretrained:
        dic = torch.load(args.pretrained)
        model.load_state_dict(dic['model'])

        
    # start a typical PyTorch training
    best_metric = -1
    #best_betti_distance = -1
    best_metric_epoch = -1
    #best_betti_distance_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter('./runs/'+dataconfig.DATA.DATASET+'/'+exp_name)
    for epoch in tqdm(range(last_epoch, config.TRAIN.MAX_EPOCHS)):
        #print("-" * 10)
        #print(f"epoch {epoch + 1}/{max_epoch}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            #print('input')
            #print(inputs.size())
            #print(torch.squeeze(inputs).permute(0,3,1,2).size())
            #print(labels.size())
            if dataconfig.DATA.IN_CHANNELS == 1:
                outputs = model(inputs)
            elif dataconfig.DATA.IN_CHANNELS == 3:
                outputs = model(torch.squeeze(inputs).permute(0,3,1,2))
            #print(outputs.size())
            if config.LOSS.USE_LOSS == 'Dice':
                loss = loss_function(outputs, labels)
            else:
                loss, dic = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            if step % config.TRAIN.LOG_INTERVAL == 0:
                writer.add_scalar('train_loss', loss.item(), epoch_len * epoch + step)
                if config.LOSS.USE_LOSS == 'Dice':
                    writer.add_scalar('dice', loss.item(), epoch_len * epoch + step)
                else:
                    for key, val in dic.items():
                        writer.add_scalar(key, val.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        #print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % config.TRAIN.VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                #betti_distances = []
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    roi_size = tuple(dataconfig.DATA.IMG_SIZE)
                    sw_batch_size = 4
                    if dataconfig.DATA.IN_CHANNELS == 1:
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    elif dataconfig.DATA.IN_CHANNELS == 3:
                        val_outputs = sliding_window_inference(torch.squeeze(val_images).permute(0,3,1,2), roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    #for pair in zip(val_outputs,val_labels):
                    #    betti_distances.append(compute_Betti_distance(pair))
                #betti_distance = torch.mean(torch.stack(betti_distances).float())
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                dic = {}
                dic['model'] = model.state_dict()
                dic['optimizer'] = optimizer.state_dict()
                dic['scheduler'] = scheduler.state_dict()
                torch.save(dic, './models/'+dataconfig.DATA.DATASET+'/'+exp_name+'/last_model_dict.pth')
                if (epoch+1)%(int(config.TRAIN.MAX_EPOCHS/10)) == 0:
                    torch.save(dic, './models/'+dataconfig.DATA.DATASET+'/'+exp_name+'/epoch'+str(epoch+1)+'_model_dict.pth')
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(dic, './models/'+dataconfig.DATA.DATASET+'/'+exp_name+'/best_model_dict.pth')
                
                #if betti_distance < best_betti_distance:
                #    best_betti_distance = betti_distance
                #    best_betti_distance_epoch = epoch + 1
                #    torch.save(dic, 'best_betti_model_'+name+'_dict.pth')
                    #print("saved new best metric model")
                #print(
                #    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                #        epoch + 1, metric, best_metric, best_metric_epoch
                #    )
                #)
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                #writer.add_scalar("val_mean_betti", betti_distance, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                if dataconfig.DATA.IN_CHANNELS == 1:
                    plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                elif dataconfig.DATA.IN_CHANNELS == 3:
                    plot_2d_or_3d_image(torch.squeeze(val_images).permute(0,3,1,2), epoch + 1, writer, index=0, tag="image", max_channels=3)
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    #print(f"train completed, best_betti_distance: {best_betti_distance:.4f} at epoch: {best_betti_distance_epoch}")
    writer.close()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    main(args)