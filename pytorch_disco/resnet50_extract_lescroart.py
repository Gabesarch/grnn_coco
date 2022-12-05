import os
# script = """
MODE="CLEVR_STA"
# export MODE
# """
# os.system("bash -c '%s'" % script)
os.environ["MODE"] = MODE
os.environ["exp_name"] = 'replica_multiview_trainer_pretrain'
os.environ["run_name"] = 'grnn'

import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
# from backend import saverloader, inputs
from sklearn.decomposition import PCA

from model_base import Model
# from nets.featnet2D import FeatNet2D
# from nets.feat3dnet import Feat3dNet
# from nets.emb3dnet import Emb3dNet
# from nets.occnet import OccNet
# # from nets.mocnet import MocNet
# # from nets.viewnet import ViewNet
# from nets.rgbnet import RgbNet
# from nets.featnet import FeatNet


import cv2

from tensorboardX import SummaryWriter
import torch.nn.functional as F

import utils_samp
import utils_geom
import utils_improc
import utils_basic
import utils_eval
# import utils_py
import utils_misc
# import utils_track
import utils_vox_other as utils_vox

from torchvision import transforms
from torchvision import datasets
import torchvision

import matplotlib.pyplot as plt
from matplotlib import cm
# import seaborn as sn

from pycocotools.coco import COCO

from tqdm import tqdm


# from backend import saverloader, inputs
from torchvision import datasets, models, transforms


import sys
# sys.path.append("XTConsistency")
# from modules.unet import UNet, UNetReshade

np.set_printoptions(precision=2)
np.random.seed(0)

import ipdb
st = ipdb.set_trace

import dino.dino_utils

sys.path.append("dino")
import vision_transformer as vits
import colorsys
import random
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from vision_transformer import DINOHead

do_feat3d = True

XTC_init = '/home/gsarch/repo/pytorch_disco/saved_checkpoints/XTC_checkpoints/rgb2depth_consistency_wimagenet.pth'
lescroart_images_root_path = '/lab_data/tarrlab/common/datasets/Lescroart2018_fmri/'

set_name = 'dino_markdata_300epoch_resnet50'

set_name = 'simclr_markdata_300epoch_resnet50'
set_name = 'resnet50_IN_markdata_300epoch_resnet50'
# model_load = 'simclr'
model_load = 'resnet50_trained_on_IN'
batch_size = 1

tag_folders = {
    'trn1': '/lab_data/tarrlab/common/datasets/Lescroart2018_fmri/stimuli_trn_run0/',
    'trn2': '/lab_data/tarrlab/common/datasets/Lescroart2018_fmri/stimuli_trn_run1/',
    'trn3': '/lab_data/tarrlab/common/datasets/Lescroart2018_fmri/stimuli_trn_run2/',
    'trn4': '/lab_data/tarrlab/common/datasets/Lescroart2018_fmri/stimuli_trn_run3/',
    'test': '/lab_data/tarrlab/common/datasets/Lescroart2018_fmri/stimuli_val/',
    }

tag_sizes = {
    'trn1': 9000,
    'trn2': 9000,
    'trn3': 9000,
    'trn4': 9000,
    'test': 2700,
    }

layers = [
    'out_initial',
    # 'out_1_0',
    # 'out_1_1',
    'out_1_2',
    # 'out_2_0',
    # 'out_2_1',
    # 'out_2_2',
    'out_2_3',
    # 'out_3_0',
    # 'out_3_1',
    # 'out_3_2',
    # 'out_3_3',
    # 'out_3_4',
    'out_3_5',
    # 'out_4_0',
    # 'out_4_1',
    'out_4_2',
    'out_avgpool',
]

pretrain = True

checkpoint_dir='checkpoints/' + set_name
log_dir='logs_grnn_coco'

save_feats = True
plot_classes = False
only_process_stim_ids = False
log_freq = 1000 # frequency for logging tensorboard
# subj = 1 # subject number - supports: 1, 2, 7 - only for when only_process_stim_ids=True
block_average = True
block_average_len = 30 # 30 stimulus frames per TR
zscore_after_block_averaging = False # z score features after block averaging


if only_process_stim_ids:
    stim_list = np.load(
        "/user_data/yuanw3/project_outputs/NSD/output/coco_ID_of_repeats_subj%02d.npy" % (subj)
    )
    stim_list = list(stim_list)

output_dir = f'/lab_data/tarrlab/gsarch/encoding_model/{set_name}'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

print("set name:", set_name)


# for plotting attention masks
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    # fig.savefig(fname)
    # print(f"{fname} saved.")
    return fig

# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, coco_images_path, transform):
        super(ImageFolderWithPaths, self).__init__(coco_images_path, transform=transform)
        order = []
        for file_ in self.imgs:
            order.append(int(file_[0][-11:-4]))
        # imgs = np.array(self.imgs)
        sorted_inds = np.argsort(np.array(order))
        # self.imgs = list(imgs[np.argsort(np.array(order))])
        self.imgs = [self.imgs[j] for j in list(sorted_inds)]

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        path = path[-11:-4]
        file_id = int(path)

        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (file_id,))
        return tuple_with_path

class RESNET(nn.Module):
    def __init__(self):
        super(RESNET, self).__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model_urls = {
            'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
            'alexnet_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
        }

        pretrained = False
        if model_load=='dino':
            path = '/user_data/gsarch/dino_output_300epoch_lescroart_resnet50/checkpoint.pth'
            arch = "resnet50"
            # checkpoint = torch.load(path)
            checkpoint_key = "teacher"
            # state_dict = checkpoint[key_]
            # # remove `module.` prefix
            # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # # remove `backbone.` prefix induced by multicrop wrapper
            # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            # checkpoint["state_dict"] = state_dict
            model = models.__dict__[arch](num_classes=0)
            model.fc = nn.Identity()
            # dino.dino_utils.load_pretrained_weights(model, path, "teacher", arch, patch_size)
            state_dict = torch.load(path, map_location="cpu")
            epoch = state_dict["epoch"]
            print(f"Loaded from epoch: {epoch}")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(path, msg))
        elif model_load=='simclr':
            path = '/user_data/gsarch/simclr_output_300epoch_lescroart/checkpoint_0300.pth.tar'
            print(path)
            checkpoint = torch.load(path)
            state_dict = checkpoint["state_dict"]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            # new_state_dict = {}
            # for k, v in state_dict.items():
            #     name = k[16:] # remove `module.`
            #     new_state_dict[name] = v
            checkpoint["state_dict"] = state_dict
        elif model_load=='resnet50_trained_on_IN':
            pretrained = True
        elif model_load=='resnet50_trained_on_SIN':
            checkpoint = model_zoo.load_url(model_urls[model_load])
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            checkpoint['state_dict'] = state_dict
        elif model_load=='resnet50_trained_on_SIN_and_IN':
            checkpoint = model_zoo.load_url(model_urls[model_load])
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            checkpoint['state_dict'] = state_dict
        elif model_load=='resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN':
            checkpoint = model_zoo.load_url(model_urls[model_load])
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            checkpoint['state_dict'] = state_dict
        else:
            assert(False) # dont know this model

        if model_load=='simclr_replica_carla':
            out_dim = 128
            model = models.resnet50(pretrained=False, num_classes=out_dim)
            dim_mlp = model.fc.in_features
            # add mlp projection head
            model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
        elif model_load=='dino_replica_carla':
            model = models.resnet50(pretrained=pretrained)
            embed_dim = model.fc.weight.shape[1]
            out_dim = 65536
            use_bn_in_head = False
            model = dino.utils.MultiCropWrapper(
                model,
                DINOHead(embed_dim, out_dim, use_bn_in_head),
            )
        else:
            model = models.resnet50(pretrained=pretrained)

        
        if not pretrained and not model_load=='dino':
            msg = model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("Loaded with msg:", msg)

        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        # list_children = list(model.children())
        # print(list_children)

        self.initial_layers = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        self.layer1_0 = model.layer1[0]
        self.layer1_1 = model.layer1[1]
        self.layer1_2 = model.layer1[2]
        self.layer2_0 = model.layer2[0]
        self.layer2_1 = model.layer2[1]
        self.layer2_2 = model.layer2[2]
        self.layer2_3 = model.layer2[3]
        self.layer3_0 = model.layer3[0]
        self.layer3_1 = model.layer3[1]
        self.layer3_2 = model.layer3[2]
        self.layer3_3 = model.layer3[3]
        self.layer3_4 = model.layer3[4]
        self.layer3_5 = model.layer3[5]
        self.layer4_0 = model.layer4[0]
        self.layer4_1 = model.layer4[1]
        self.layer4_2 = model.layer4[2]
        # self.layer1 = model.layer1
        # self.layer2 = model.layer2
        # self.layer3 = model.layer3
        # self.layer4 = model.layer4
        self.avgpool = model.avgpool
        if model_load in ["simclr_replica_carla", "dino_replica_carla"]:
            self.fc1 = model.fc[0]
            self.fc2 = model.fc[1:]

        self.transform = transforms.Compose([
            # transforms.Resize((480, 480)),
            # transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.model_name = model_load

        self.B = batch_size

        self.layers_keep = layers

        # self.avgpool2d = nn.AvgPool2d(2, stride=2)


    def forward(self, x):
        """Forward pass.
        Args:
            x (PIL): input data (image)
        Returns:
            tensor: depth
        """

        x = self.transform(x)

        out_initial = self.initial_layers(x.to(self.device))
        # layer1_out = self.layer1(out0)
        # layer2_out = self.layer2(layer1_out)
        # layer3_out = self.layer3(layer2_out)
        # layer4_out = self.layer4(layer3_out)
        # avgpool_out = self.avgpool(layer4_out)
        out_1_0 = self.layer1_0(out_initial)
        out_1_1 = self.layer1_1(out_1_0)
        out_1_2 = self.layer1_2(out_1_1) 
        out_2_0 = self.layer2_0(out_1_2)
        out_2_1 = self.layer2_1(out_2_0)
        out_2_2 = self.layer2_2(out_2_1)
        out_2_3 = self.layer2_3(out_2_2)
        out_3_0 = self.layer3_0(out_2_3)
        out_3_1 = self.layer3_1(out_3_0)
        out_3_2 = self.layer3_2(out_3_1)
        out_3_3 = self.layer3_3(out_3_2)
        out_3_4 = self.layer3_4(out_3_3)
        out_3_5 = self.layer3_5(out_3_4)
        out_4_0 = self.layer4_0(out_3_5)
        out_4_1 = self.layer4_1(out_4_0)
        out_4_2 = self.layer4_2(out_4_1)
        avgpool_out = self.avgpool(out_4_2)
        avgpool_out = avgpool_out.view(self.B, -1)
        if self.model_name in ["simclr_replica_carla", "dino_replica_carla"]:
            fc1_out = self.fc1(avgpool_out)
            fc2_out = self.fc2(fc1_out)

        layers = {}
        layers['out_initial'] = out_initial
        layers['out_1_0'] = out_1_0
        layers['out_1_1'] = out_1_1
        layers['out_1_2'] = out_1_2
        layers['out_2_0'] = out_2_0
        layers['out_2_1'] = out_2_1
        layers['out_2_2'] = out_2_2
        layers['out_2_3'] = out_2_3
        layers['out_3_0'] = out_3_0
        layers['out_3_1'] = out_3_1
        layers['out_3_2'] = out_3_2
        layers['out_3_3'] = out_3_3
        layers['out_3_4'] = out_3_4
        layers['out_3_5'] = out_3_5
        layers['out_4_0'] = out_4_0
        layers['out_4_1'] = out_4_1
        layers['out_4_2'] = out_4_2
        # layers['layer1_out'] = layer1_out
        # layers['layer2_out'] = layer2_out
        # layers['layer3_out'] = layer3_out
        # layers['layer4_out'] = layer4_out
        layers['out_avgpool'] = avgpool_out
        if self.model_name in ["simclr_replica_carla", "dino_replica_carla"]:
            layers['out_fc1'] = fc1_out
            layers['out_fc2'] = fc2_out

        # only return layers we want to keep
        layers_ = {}
        for lk in self.layers_keep:
            layers_[lk] = layers[lk]

        return layers_


class CARLA_MOC(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = COCO_GRNN()

        self.model.eval()

        # self.start_iter = saverloader.load_weights(self.model, None)

        self.model.run_extract()
        # self.model.plot_tsne_only()
        # self.model.plot_depth()

class COCO_GRNN(nn.Module):
    def __init__(self):
        super(COCO_GRNN, self).__init__()

        # path = '/home/gsarch/repo/3DQNets/pytorch_disco/checkpoints/02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_train_ns_grnn00/model-60000.pth'
        # checkpoint = torch.load(path)
        # self.feat3dnet.load_state_dict(checkpoint['model_state_dict'])
        if plot_classes:
            COCO_super_cat = ["person", "vehicle", "outdoor", "animal", "accessory", "sports", "kitchen", "food", "furtniture", "electronics", "appliance", "indoor"]
            COCO_cat = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
                'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]

            self.COCO_cat_NSD = np.load(coco_images_path + '/NSD_cat_feat.npy')
            self.COCO_supcat_NSD = np.load(coco_images_path + '/NSD_supcat_feat.npy')
            self.coco_util_train = COCO(coco_images_path + '/annotations/instances_train2014.json')
            self.coco_util_val = COCO(coco_images_path + '/annotations/instances_val2014.json')

        # midas but remove last layer
        self.resnet = RESNET() #torch.nn.Sequential(*(list(midas.pretrained.children())+list(midas.scratch.children()))[:-1])
        self.resnet.cuda()
        self.resnet.eval()
        # print(self.resnet)

        # self.W = 256
        # self.H = 256
        # self.W = 384
        # self.H = 384
        # self.fov = 60
        # hfov = float(self.fov) * np.pi / 180.
        # self.pix_T_camX = np.array([
        #     [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
        #     [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
        #     [0., 0.,  1, 0],
        #     [0., 0., 0, 1]])
        # self.pix_T_camX[0,2] = self.W/2. 
        # self.pix_T_camX[1,2] = self.H/2. 

        self.tags = ['trn1', 'trn2', 'trn3', 'trn4', 'test']

        self.B = batch_size
        assert(self.B==1) 
        # self.pix_T_camX = torch.from_numpy(self.pix_T_camX).cuda().unsqueeze(0).repeat(self.B,1,1).float()

        # data_loader_transform = transforms.Compose([
        #                     transforms.ToTensor(),
        #                     ])
        # dataset = ImageFolderWithPaths(coco_images_path, transform=data_loader_transform) # our custom dataset
        
        # # self.dataloader = torch.utils_DataLoader(dataset, batch_size=32, shuffle=False)
        # # dataset = datasets.ImageFolder(coco_images_path, transform=transform)
        # self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.B, shuffle=False)

        self.writer = SummaryWriter(log_dir + '/' + set_name, max_queue=10, flush_secs=1000)

        self.toPIL = transforms.ToPILImage()

        self.pool_len = 2
        self.pool3d = nn.AvgPool3d(self.pool_len, stride=self.pool_len)
        self.pool2d = nn.AvgPool2d(self.pool_len, stride=self.pool_len)

    
    def run_extract(self):

        for tag in tqdm(self.tags, leave=True):
            # layer_save_name = f'{layer}_{tag}'
            # if os.path.isfile(f'{output_dir}/{layer_save_name}.npy'):
            #     print("LAYER ALREADY EXISTS... SKIPPING")
            #     continue

            feature_size = tag_sizes[tag]
            image_folder = tag_folders[tag]

            # reinitialize dataloader for this tag
            data_loader_transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.ToTensor(),
                            ])
            dataset = ImageFolderWithPaths(image_folder, transform=data_loader_transform) # our custom dataset
            
            # self.dataloader = torch.utils_DataLoader(dataset, batch_size=32, shuffle=False)
            # dataset = datasets.ImageFolder(coco_images_path, transform=transform)
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.B, shuffle=False)

            feats_all = {}
            # for layer in layers:
            #     feats_all[layer] = np.zeros((feature_size, 384)).astype(np.float32)

            # out0_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # out1_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # out2_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # out3_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # out4_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # out5_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # out6_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # out7_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # out8_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # out9_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # out10_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # out11_feats = np.zeros((feature_size, 384)).astype(np.float32)
            # attention = np.zeros((73000, 4056)).astype(np.float32)
            # file_order = np.zeros(73000)

            cat_names = []
            supcat_names = []
            cat_ids = []
            idx = 0
            for images, _, file_ids in tqdm(self.dataloader, leave=False):

                if only_process_stim_ids:
                    if file_ids not in stim_list: 
                        continue

                # print('Images processed: ', idx)

                self.summ_writer = utils_improc.Summ_writer(
                    writer=self.writer,
                    global_step=idx,
                    set_name=set_name,
                    log_freq=log_freq,
                    fps=8,
                    # just_gif=True,
                )

                # assert(images.shape[2]==424)
                # assert(images.shape[3]==424)
                assert(images.shape[0]==1)
                # images = images.squeeze(0)
                # images_pil = self.toPIL(images).convert('RGB')

                if False: #self.summ_writer.save_this:
                    rgb_camX = images.cuda().float().unsqueeze(0)
                    rgb_camX_norm = rgb_camX - 0.5
                    self.summ_writer.summ_rgb('inputs/rgb', rgb_camX_norm)
                    del rgb_camX, rgb_camX_norm

                if False:
                    plt.figure()
                    rgb_camXs_np = rgb_camXs[0].permute(1,2,0).detach().cpu().numpy()
                    plt.imshow(rgb_camXs_np)
                    plt.savefig('images/test.png')


                # if plot_classes:
                #     # get category name
                #     for b in range(self.B):
                #         img_id = [int(file_ids[b].detach().cpu().numpy())]
                #         coco_util = self.coco_util_train
                #         annotation_ids = coco_util.getAnnIds(img_id)
                #         if not annotation_ids:
                #             coco_util = self.coco_util_val
                #             annotation_ids = coco_util.getAnnIds(img_id)
                #         annotations = coco_util.loadAnns(annotation_ids)

                #         best_area = 0
                #         entity_id = None
                #         entity = None
                #         for i in range(len(annotations)):
                #             if annotations[i]['area'] > best_area:
                #                 entity_id = annotations[i]["category_id"]
                #                 entity = coco_util.loadCats(entity_id)[0]["name"]
                #                 super_cat = coco_util.loadCats(entity_id)[0]["supercategory"]
                #                 best_area = annotations[i]['area']
                #         cat_names.append(entity)
                #         cat_ids.append(entity_id)
                #         supcat_names.append(super_cat)
                
                # estimate depth
                out_dict = self.resnet(images)

                

                if save_feats:
                    for layer in layers:
                        # feats_all[layer][idx] = out_dict[layer].squeeze().detach().cpu().numpy() 
                        feat_memX = out_dict[layer]

                        num_spatial = len(feat_memX.shape[2:])
                        feat_size_flat = np.prod(torch.tensor(feat_memX.shape[1:]).numpy())

                        feat_size_thresh = 200000

                        while feat_size_flat>feat_size_thresh:
                            if num_spatial==3:
                                feat_memX = self.pool3d(feat_memX)
                            elif num_spatial==2:
                                feat_memX = self.pool2d(feat_memX)
                            elif num_spatial==1:
                                break
                            elif num_spatial==0:
                                break
                            else:
                                assert(False)
                            feat_size_flat = np.prod(torch.tensor(feat_memX.shape[1:]).numpy())

                        feat_memX = feat_memX.cpu().numpy().astype(np.float32)
                        # print(feat_memX.shape)

                        if idx==0:
                            # print("Initializing layer", layer_)
                            feature_size = tag_sizes[tag]
                            if num_spatial==3:
                                b,c,h,w,d = feat_memX.shape
                                feats_all[layer] = np.zeros((feature_size, c, h, w, d), dtype=np.float32)
                            elif num_spatial==2:
                                b,c,h,w = feat_memX.shape
                                feats_all[layer] = np.zeros((feature_size, c, h, w), dtype=np.float32)
                                # file_order = np.zeros(73000)
                            elif num_spatial==1:
                                b,c,h = feat_memX.shape
                                feats_all[layer] = np.zeros((feature_size, c, h), dtype=np.float32)
                                # file_order = np.zeros(73000)
                            elif num_spatial==0:
                                b,c = feat_memX.shape
                                feats_all[layer] = np.zeros((feature_size, c), dtype=np.float32)
                                # file_order = np.zeros(73000)
                            else:
                                assert(False)

                        

                        feats_all[layer][idx] = feat_memX

                    # attention_ = attention_.reshape(self.B, -1)
                    # out0_feats[idx] = out0.squeeze().detach().cpu().numpy() 
                    # out1_feats[idx] = out1.squeeze().detach().cpu().numpy() 
                    # out2_feats[idx] = out2.squeeze().detach().cpu().numpy() 
                    # out3_feats[idx] = out3.squeeze().detach().cpu().numpy() 
                    # out4_feats[idx] = out4.squeeze().detach().cpu().numpy() 
                    # out5_feats[idx] = out5.squeeze().detach().cpu().numpy() 
                    # out6_feats[idx] = out6.squeeze().detach().cpu().numpy() 
                    # out7_feats[idx] = out7.squeeze().detach().cpu().numpy() 
                    # out8_feats[idx] = out8.squeeze().detach().cpu().numpy() 
                    # out9_feats[idx] = out9.squeeze().detach().cpu().numpy() 
                    # out10_feats[idx] = out10.squeeze().detach().cpu().numpy() 
                    # out11_feats[idx] = out11.squeeze().detach().cpu().numpy() 
                    # attention[idx] = attention_.detach().cpu().numpy() 
                    # file_order[idx] = file_ids.detach().cpu().numpy() 

                if False: #self.summ_writer.save_this:
                    nh = attention_.shape[1] # number of head

                    # we keep only the output patch attention
                    attention_ = attention_[0, :, 0, 1:]#.reshape(nh, -1)
                    # nh = attention_.shape[1]
                    attentions = attention_.reshape(nh, -1)
                    threshold = 0.6

                    if threshold is not None:
                        w_featmap = images.shape[-2] // 16
                        h_featmap = images.shape[-1] // 16
                        # we keep only a certain percentage of the mass
                        val, idx3 = torch.sort(attentions)
                        val /= torch.sum(val, dim=1, keepdim=True)
                        cumval = torch.cumsum(val, dim=1)
                        th_attn = cumval > (1 - threshold)
                        idx2 = torch.argsort(idx3)
                        for head in range(nh):
                            th_attn[head] = th_attn[head][idx2[head]]
                        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                        # interpolate
                        # th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
                        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), size=(images.shape[-2], images.shape[-1]), mode="nearest")[0].cpu().numpy()
                        

                        attentions = attentions.reshape(nh, w_featmap, h_featmap)
                        # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
                        attentions = nn.functional.interpolate(attentions.unsqueeze(0), size=(images.shape[-2], images.shape[-1]), mode="nearest")[0].cpu().numpy()

                        output_dir_vis = 'dino/visuals'
                        # save attentions heatmaps
                        os.makedirs(output_dir_vis, exist_ok=True)
                        torchvision.utils.save_image(torchvision.utils.make_grid(images, normalize=True, scale_each=True), os.path.join(output_dir_vis, "img.png"))
                        for j in range(nh):
                            # fname = os.path.join(output_dir, "attn-head" + str(j) + "_idx=" + str(idx) + ".png")
                            fname = "attn-head" + str(j)
                            fig =plt.figure()
                            plt.imshow(attentions[j])
                            self.summ_writer.summ_figure('attention/' + fname, fig)
                            # plt.imsave(fname=fname, arr=attentions[j], format='png')
                            print(f"{fname} saved.")

                        if threshold is not None:
                            # image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
                            for j in range(nh):
                                fig = display_instances(images.permute(1,2,0).cpu().numpy(), th_attn[j], fname=os.path.join(output_dir_vis, "mask_th" + str(threshold) + "_head" + str(j) + "_idx=" + str(idx) + ".png"), blur=False)
                                self.summ_writer.summ_figure('attention/' + "mask_th" + str(threshold) + "_head" + str(j), fig)

                            

                idx += 1*self.B

                # plt.close('all')

                # if idx == 10:
                #     assert(False)

                # if only_process_stim_ids:
                #     if idx == 10000:
                #         break
            
            # feats = np.concatenate(feats, axis=0)
            # file_order = np.concatenate(file_order, axis=0)
            # dim_red = False
            # if dim_red:
            #     pca = PCA(n_components=1000)

            #     b,c,h,w,d = feats.shape
            #     feats = np.reshape(feats, (b, -1))
            #     feats = pca.fit_transform(feats)

            #     plt.plot(np.cumsum(pca.explained_variance_ratio_))
            #     plt.xlabel('number of components')
            #     plt.ylabel('cumulative explained variance')
            #     plt.yticks(np.arange(0.0,1.0,0.05))
            #     plt.savefig(output_dir + '/variance_explained.png')

            if save_feats:
                for layer in layers:
                    feats = feats_all[layer]
                    if block_average:
                        print("Block averaging...")
                        cur_feat_len = feats.shape[0]
                        new_feat_len = cur_feat_len/block_average_len
                        assert(new_feat_len.is_integer())
                        new_feat_len = int(new_feat_len)
                        indices_blocks = np.arange(0, cur_feat_len+1, block_average_len)
                        new_dims = list(feats.shape[1:])
                        new_dims = [new_feat_len] + new_dims
                        feats_ = np.zeros(new_dims)
                        for ii in range(new_feat_len):
                            feats_[ii] = np.mean(feats[indices_blocks[ii]:indices_blocks[ii+1]], axis=0)
                        if zscore_after_block_averaging:
                            feats_ = scipy.stats.zscore(feats_, axis=0)
                        feats = feats_
                        print("new feature length is:", feats.shape)
                    np.save(f'{output_dir}/{layer}_{tag}.npy', feats)
                # np.save(f'{output_dir}/out0_feats.npy', out0_feats)
                # np.save(f'{output_dir}/out1_feats.npy', out1_feats)
                # np.save(f'{output_dir}/out2_feats.npy', out2_feats)
                # np.save(f'{output_dir}/out3_feats.npy', out3_feats)
                # np.save(f'{output_dir}/out4_feats.npy', out4_feats)
                # np.save(f'{output_dir}/out5_feats.npy', out5_feats)
                # np.save(f'{output_dir}/out6_feats.npy', out6_feats)
                # np.save(f'{output_dir}/out7_feats.npy', out7_feats)
                # np.save(f'{output_dir}/out8_feats.npy', out8_feats)
                # np.save(f'{output_dir}/out9_feats.npy', out9_feats)
                # np.save(f'{output_dir}/out10_feats.npy', out10_feats)
                # np.save(f'{output_dir}/out11_feats.npy', out11_feats)
                # np.save(f'{output_dir}/attention.npy', attention)
                # np.save(f'{output_dir}/file_order.npy', file_order)
                # if plot_classes:
                #     np.save(f'{output_dir}/supcat.npy', np.array(supcat_names))
                #     np.save(f'{output_dir}/cat.npy', np.array(cat_names))

        # if plot_classes:
        #     feats = np.reshape(feats, (feats.shape[0], -1))

        #     tsne = TSNE(n_components=2).fit_transform(feats)
        #     # pred_catnames_feats = [self.maskrcnn_to_ithor[i] for i in self.feature_obj_ids]

        #     self.plot_by_classes(supcat_names, tsne, self.summ_writer)

        #     # tsne plot colored by predicted labels
        #     tsne_pred_figure = self.get_colored_tsne_image(supcat_names, tsne)
        #     self.summ_writer.summ_figure(f'tsne/tsne_grnn_reduced_supcat', tsne_pred_figure)

        #     tsne_pred_figure = self.get_colored_tsne_image(cat_names, tsne)
        #     self.summ_writer.summ_figure(f'tsne/tsne_grnn_reduced_cat', tsne_pred_figure)


if __name__ == '__main__':
    model = CARLA_MOC(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    model.initialize_model()