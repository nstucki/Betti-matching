from BettiMatching import *
from skimage.transform import rescale
import yaml
import os
import json
from os.path import join
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
from glob import glob
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import monai
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AddChanneld, AsDiscrete, Compose, LoadImaged, SaveImage, ScaleIntensityd, EnsureTyped, EnsureType
from monai.networks.nets import UNet
import torchvision
import imageio
from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np
from sklearn.metrics import accuracy_score
from metrics.rand import adapted_rand
from metrics.voi import voi

import pandas as pd

from tqdm import tqdm

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

parser = ArgumentParser()
parser.add_argument('--folder',
                    default=None,
                    help='root folder of all the models')
parser.add_argument('--config',
                    default=None,
                    help='config file (.yaml) containing the hyper-parameters for training.')
parser.add_argument('--dataconfig',
                    default=None,
                    help='data config file (.yaml) containing the dataset specific information.')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0],
                        help='list of index where skip conn will be made')
parser.add_argument('--metrics',
                    default='',
                    help='metrics to compute (comma separated list of metrics)')


def Dice(prediction, ground_truth):
    dice = np.sum(prediction[ground_truth==1])*2.0 / (np.sum(prediction) + np.sum(ground_truth))
    return dice


def Accuracy(prediction, ground_truth):
    m,n = prediction.shape
    acc = np.sum(prediction==ground_truth) / (m*n)
    return acc


def normalize(Picture, scale=1, anti_aliasing=True):
    Picture = rescale(Picture, scale=scale, anti_aliasing=anti_aliasing)
    Picture = Picture / np.max(Picture)
    return Picture


def load_model(path, spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2):
    device = torch.device("cpu")
    model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        ).to(device)
    model.load_state_dict(torch.load(os.getcwd()+'/'+path, map_location=torch.device('cpu'))['model'])
    model.eval()
    return model


def plot_evaluation(image, models=[], segment=None, data_path='./data/cremi'):
    img = imageio.imread(data_path+'/images/image_'+str(image)+'.png')
    seg = imageio.imread(data_path+'/labels/label_'+str(image)+'.png')
    seg = np.array(seg, dtype=float)
    input = torchvision.transforms.functional.to_tensor(np.array(img))
    input = input.unsqueeze(0)[:,:,:304,:304]
    seg = seg[:304,:304]/np.max(seg)
    if segment is not None:
        img = img[segment[0][0]:segment[0][1],segment[1][0]:segment[1][1]]
        seg = seg[segment[0][0]:segment[0][1],segment[1][0]:segment[1][1]]
    outputs = []
    outputs_bin = []
    for model in models:
        output = model(input)
        output = torch.squeeze(output)
        output = torch.sigmoid(output).detach().numpy()
        if segment is not None:
            output = output[segment[0][0]:segment[0][1],segment[1][0]:segment[1][1]]
        output_bin = ((output>0.5)*1.0)
        outputs.append(output)
        outputs_bin.append(output_bin)

    fig = plt.figure(figsize=(25,25))
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.01)
    columns = len(models) + 1
    rows = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Picture')
    for i,output in enumerate(outputs):
        fig.add_subplot(rows, columns, i+2)
        plt.imshow(output, cmap='gray')
        plt.axis('off')
        plt.title('Output'+str(i))
    fig.add_subplot(rows, columns, len(models)+2)
    plt.imshow(seg, cmap='gray')
    plt.axis('off')
    plt.title('Ground Truth')
    for i,output_bin in enumerate(outputs_bin):
        fig.add_subplot(rows, columns, len(models)+3+i)
        plt.imshow(output_bin, cmap='gray')
        plt.title('Prediction'+str(i))
        plt.axis('off')
    plt.show()
    metrics = pd.DataFrame(columns=['Acc', 'Dice', 'BNerr', 'BMerr', 'BNerr 0', 'BMerr 0', 'BNerr 1', 'BMerr 1'])
    for i,output_bin in enumerate(outputs_bin):
        BM = BettiMatching(output_bin, seg, filtration='superlevel')
        metrics.loc['Model '+str(i)] = {'Acc': Accuracy(output_bin, seg), 'Dice': Dice(output_bin, seg), 'Betti': BM.Betti_number_error(), 'Betti 0': BM.Betti_number_error(dimensions=[0]), 'Betti 1': BM.Betti_number_error(dimensions=[1]), 'TM': BM.loss(), 'TM 0': BM.loss(dimensions=[0]), 'TM 1': BM.loss(dimensions=[1])}
    print(metrics)
    return


def compute_metrics(t, relative=False, comparison='union', filtration='superlevel', construction='V'):
    BM = BettiMatching(t[0], t[1], relative=relative, comparison=comparison, filtration=filtration, construction=construction)
    return BM.loss(dimensions=[0,1]), BM.loss(dimensions=[0]), BM.loss(dimensions=[1]), BM.Betti_number_error(threshold=0.5, dimensions=[0,1]), BM.Betti_number_error(threshold=0.5, dimensions=[0]), BM.Betti_number_error(threshold=0.5, dimensions=[1])


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]
    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]
    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)


def evaluation(model, dataconfig, metrics=[], save_path=None, relative=True, comparison='union', filtration='superlevel', pixel_dimension=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data_path = dataconfig.DATA.DATA_PATH
    images = sorted(glob(os.path.join(data_path+'images', "*"+dataconfig.DATA.FORMAT)))
    segs = sorted(glob(os.path.join(data_path+'labels', "*"+dataconfig.DATA.FORMAT)))
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-dataconfig.DATA.VAL_SAMPLES:], segs[-dataconfig.DATA.VAL_SAMPLES:])]
    val_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                AddChanneld(keys=["img", "seg"]),
                ScaleIntensityd(keys=["img", "seg"]),
                EnsureTyped(keys=["img", "seg"]),
            ]
        )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=4, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    #saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg")
    with torch.no_grad():
        if not os.path.exists(save_path):
            metrics_dic = {}
        else:
            metrics_dic = np.load(save_path, allow_pickle=True).item()
            if type(metrics_dic) != dict:
                metrics_dic = {}
                metrics = []
            else:
                remove_list = ['dice', 'dice std', 'cldice', 'cldice std', 'accuracy', 'accuracy std',
                                'betti error', 'betti error std',
                                'betti_0 error', 'betti_0 error std', 'betti_1 error',
                                'betti_1 error std', 'ari', 'ari std', 'voi', 'voi std']
                [metrics_dic.pop(key, None) for key in remove_list]
        losses = []
        losses_0 = []
        losses_1 = []
        betti_errors = []
        betti_0_errors = []
        betti_1_errors = []
        cldices = []
        accuracies = []
        aris = []
        vois = []
        vois_ignore_0 = []
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = tuple(dataconfig.DATA.IMG_SIZE)
            sw_batch_size = 16
            if dataconfig.DATA.IN_CHANNELS == 1:
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            elif dataconfig.DATA.IN_CHANNELS == 3:
                val_outputs = sliding_window_inference(torch.squeeze(val_images).permute(0,3,1,2), roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i).cpu() for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels.cpu())
            # compute metric for current iteration
            if (len(metrics)==0 or 'Dice' in metrics) and 'Dice' not in metrics_dic.keys():
                dice_metric(y_pred=val_outputs, y=val_labels)
            for pair in zip(val_outputs,val_labels):
                if (len(metrics) == 0 or 'ClDice' in metrics) and 'ClDice' not in metrics_dic.keys():
                    cldice = clDice(pair[0].numpy(),pair[1].numpy())
                    cldices.append(cldice)
                if (len(metrics) == 0 or 'Betti matching error' in metrics or 'Betti number error' in metrics) and ('Betti matching error' not in metrics_dic.keys() or 'Betti number error' not in metrics_dic.keys()):
                    loss, loss_0, loss_1, betti_err, betti_0_err, betti_1_err = compute_metrics(pair, relative=relative, comparison=comparison, filtration=filtration, construction='V')
                    losses.append(loss)
                    losses_0.append(loss_0)
                    losses_1.append(loss_1)
                    betti_errors.append(betti_err)
                    betti_0_errors.append(betti_0_err)
                    betti_1_errors.append(betti_1_err)
                if (len(metrics) == 0 or 'accuracy' in metrics) and 'Accuracy' not in metrics_dic.keys():
                    accuracy = accuracy_score(pair[0].numpy().flatten(), pair[1].numpy().flatten())
                    accuracies.append(accuracy)
                if (len(metrics) == 0 or 'ARI' in metrics) and 'ARI' not in metrics_dic.keys():
                    ari = adapted_rand(np.int32(pair[0].numpy()),np.int32(pair[1].numpy()))
                    aris.append(ari)
                if (len(metrics) == 0 or 'VOI' in metrics) and 'ARI' not in metrics_dic.keys():
                    voi_score = voi(np.int32(pair[0].numpy()),np.int32(pair[1].numpy()), ignore_groundtruth=[])
                    vois.append(voi_score)
                    voi_ignore_0 = voi(np.int32(pair[0].numpy()),np.int32(pair[1].numpy()), ignore_groundtruth=[0])
                    vois_ignore_0.append(voi_ignore_0)
            #for val_output in val_outputs:
            #    saver(val_output)
        # aggregate the final mean dice result
        if (len(metrics) == 0 or 'Dice' in metrics) and 'Dice' not in metrics_dic.keys():
            dice = dice_metric.aggregate().item()
            dice_std = torch.std(dice_metric.get_buffer()).item()
            print("Dice:", dice)
            print("Dice std", dice_std)
            metrics_dic['Dice'] = dice
            metrics_dic['Dice std'] = dice_std
        if (len(metrics) == 0 or 'ClDice' in metrics) and 'ClDice' not in metrics_dic.keys():
            Cldice = np.mean(cldices)
            Cldice_std = np.std(cldices)
            print("ClDice", Cldice)
            print("ClDice std", Cldice_std)
            metrics_dic['ClDice'] = Cldice
            metrics_dic['ClDice std'] = Cldice_std
        if (len(metrics) == 0 or 'Accuracy' in metrics) and 'Accuracy' not in metrics_dic.keys():
            Accuracy = np.mean(accuracies)
            Accuracy_std = np.std(accuracies)
            print("Accuracy", Accuracy)
            print("Accuracy std", Accuracy_std)
            metrics_dic['Accuracy'] = Accuracy
            metrics_dic['Accuracy std'] = Accuracy_std
        if (len(metrics) == 0 or 'Betti matching error' in metrics) and 'Betti matching error' not in metrics_dic.keys():
            BME = torch.mean(torch.stack(losses))
            BME_std = torch.std(torch.stack(losses))
            BME_0 = torch.mean(torch.stack(losses_0))
            BME_0_std = torch.std(torch.stack(losses_0))
            BME_1 = torch.mean(torch.stack(losses_1))
            BME_1_std = torch.std(torch.stack(losses_1))
            print("Betti matching error:", torch.squeeze(BME).item())
            print("Betti matching error std", torch.squeeze(BME_std).item())
            print("Betti matching error dim 0", torch.squeeze(BME_0).item())
            print("Betti matching error dim 0 std", torch.squeeze(BME_0_std).item())
            print("Betti matching error dim 1", torch.squeeze(BME_1).item())
            print("Betti matching error dim 1 std", torch.squeeze(BME_1_std).item())
            metrics_dic['Betti matching error'] = torch.squeeze(BME).item()
            metrics_dic['Betti matching error std'] = torch.squeeze(BME_std).item()
            metrics_dic['Betti matching error dim 0'] = torch.squeeze(BME_0).item()
            metrics_dic['Betti matching error dim 0 std'] = torch.squeeze(BME_0_std).item()
            metrics_dic['Betti matching error dim 1'] = torch.squeeze(BME_1).item()
            metrics_dic['Betti matching error dim 1 std'] = torch.squeeze(BME_1_std).item()
        if (len(metrics) == 0 or 'Betti number error' in metrics) and 'Betti number error' not in metrics_dic.keys():
            Betti_error = np.mean(betti_errors)
            Betti_error_std = np.std(betti_errors)
            Betti_0_error = np.mean(betti_0_errors)
            Betti_0_error_std = np.std(betti_0_errors)
            Betti_1_error = np.mean(betti_1_errors)
            Betti_1_error_std = np.std(betti_1_errors)
            print("Betti number error", Betti_error)
            print("Betti number error std", Betti_error_std)
            print("Betti number error dim 0", Betti_0_error)
            print("Betti number error dim 0 std", Betti_0_error_std)
            print("Betti number error dim 1", Betti_1_error)
            print("Betti number error dim 1 std", Betti_1_error_std)
            metrics_dic['Betti number error'] = Betti_error
            metrics_dic['Betti number error std'] = Betti_error_std
            metrics_dic['Betti number error dim 0'] = Betti_0_error
            metrics_dic['Betti number error dim 0 std'] = Betti_0_error_std
            metrics_dic['Betti number error dim 1'] = Betti_1_error
            metrics_dic['Betti number error dim 1 std'] = Betti_1_error_std
        if (len(metrics) == 0 or 'ARI' in metrics) and 'ARI' not in metrics_dic.keys():
            Ari = np.mean(aris)
            Ari_std = np.std(aris)
            print("ARI", Ari)
            print("ARI std", Ari_std)
            metrics_dic['ARI'] = Ari
            metrics_dic['ARI std'] = Ari_std
        if (len(metrics) == 0 or 'VOI' in metrics) and 'VOI' not in metrics_dic.keys():
            Voi = np.mean(vois)
            Voi_std = np.std(vois)
            Voi_ignore_0 = np.mean(vois_ignore_0)
            Voi_ignore_0_std = np.std(vois_ignore_0)
            print("VOI", Voi)
            print("VOI std", Voi_std)
            print("VOI ignore 0", Voi_ignore_0)
            print("VOI ignore 0 std", Voi_ignore_0_std)
            metrics_dic['VOI'] = Voi
            metrics_dic['VOI std'] = Voi_std
            metrics_dic['VOI_ignore_0'] = Voi_ignore_0
            metrics_dic['VOI_ignore_0 std'] = Voi_ignore_0_std
        np.save(save_path,metrics_dic)
    return


def load_evaluations(folder, key='', models='last'):
    assert models in ['best','last']
    path_file = os.path.dirname(__file__)+'/models/cremi'
    evaluations_df = pd.DataFrame()
    for path, subdirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".npy") and file.startswith(models):
                if key in path:
                    evaluation = np.load(os.path.join(path,file), allow_pickle=True).item()
                    evaluation_df = pd.DataFrame(evaluation, index=[os.path.relpath(os.path.join(path,file),path_file)])
                    evaluations_df = pd.concat([evaluations_df,evaluation_df])
    return evaluations_df


def main(args):
    # Load the dataconfig files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.dataconfig)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)
    with open(args.dataconfig) as f:
        print('\n*** Dataconfig file')
        print(args.dataconfig)
        dataconfig = yaml.load(f, Loader=yaml.FullLoader)
    dataconfig = dict2obj(dataconfig)
    files = os.listdir(args.folder)
    metrics = [metric for metric in args.metrics.split(',')]
    for path, subdirs, files in os.walk(args.folder):
        for file in files:
            if file.endswith(".pth") and file.startswith('last'):
                print(os.path.join(path, file))
                model = load_model(os.path.join(path, file), spatial_dims=dataconfig.DATA.DIM, in_channels=dataconfig.DATA.IN_CHANNELS, out_channels=dataconfig.DATA.OUT_CHANNELS, channels=config.MODEL.CHANNELS, strides=config.MODEL.STRIDES, num_res_units=config.MODEL.NUM_RES_UNITS)
                save_path = os.path.join(path, file)[:-3]+'npy'
                evaluation(model, dataconfig, save_path=save_path, metrics=args.metrics)
                print('-----------------------------Evaluation Done-------------------------------------')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    main(args)