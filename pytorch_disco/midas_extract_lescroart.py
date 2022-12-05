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

# from .blocks import (
#     # FeatureFusionBlock,
#     # FeatureFusionBlock_custom,
#     # Interpolate,
#     # _make_encoder,
#     forward_vit,
# )

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

set_name = 'midas_markdata'
set_name = 'midas_markdata_DPT_Large'




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
    "layer_2",
    "layer_4",
    "layer_2_rn",
    "layer_4_rn",
    "path_3",
    "path_1",
    "out",
    # "out2",
    # "out3",
]

pretrained = True
# pretrained_imagenet = True

# assert(not pretrain and pretrained_imagenet)

# pretrain_mode = "repcarl"
# pretrain_mode = "midas"

checkpoint_dir='checkpoints/' + set_name
log_dir='logs_grnn_coco'

save_feats = True
plot_classes = False
only_process_stim_ids = False
log_freq = 1000 # frequency for logging tensorboard
# subj = 1 # subject number - supports: 1, 2, 7 - only for when only_process_stim_ids=True
block_average = False
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

# class DINO(nn.Module):
#     def __init__(self):
#         super(DINO, self).__init__()

#         self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         # # build model
#         # self.model = vits.__dict__['vit_small'](patch_size=16) #, num_classes=0)
#         # for p in self.model.parameters():
#         #     p.requires_grad = False
#         # self.model.eval()
#         # self.model.to(self.device)
#         # if pretrain:
#         #     url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth" # model used for visualizations in DINO paper
#         #     state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
#             # self.model.load_state_dict(state_dict, strict=True)
            
#         # path = '/user_data/gsarch/dino_output/checkpoint0100.pth'
#         # path = '/user_data/gsarch/dino_output_300epoch/checkpoint0300.pth'
        
#         arch = "vit_small"
#         patch_size = 16

#         if "vit" in arch:
#             model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
#             print(f"Model {arch} {patch_size}x{patch_size} built.")
#         elif "xcit" in arch:
#             model = torch.hub.load('facebookresearch/xcit', arch, num_classes=0)
#         elif arch in torchvision_models.__dict__.keys():
#             model = torchvision_models.__dict__[arch](num_classes=0)
#             model.fc = nn.Identity()
#         else:
#             print(f"Architecture {arch} non supported")
#             sys.exit(1)
#         model.cuda()

#         if pretrain_mode=="imagenet":
#             # url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall16_300ep_pretrain.pth" # model used for visualizations in DINO paper
#             # state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
#             # model.load_state_dict(state_dict, strict=True)
#             vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
#             print("LOADED IMAGENET PRETRAINED")

#         elif pretrain_mode=="markdata":
#             path = '/user_data/gsarch/dino_output_300epoch_lescroart/checkpoint.pth'

#             print("PATH IS:", path)

#             dino.dino_utils.load_pretrained_weights(model, path, "teacher", arch, patch_size)

#         elif pretrain_mode=="repcarl":

#             path = '/user_data/gsarch/dino_output_300epoch/checkpoint0300.pth' # this is trained on replica & carla

#             print("PATH IS:", path)

#             # to_restore = {"epoch": 0}
#             # dino.utils.restart_from_checkpoint(
#             #     path,
#             #     run_variables=to_restore,
#             #     teacher=model,
#             # )
#             # st()
#             # model = dino.utils.MultiCropWrapper(
#             #     model,
#             #     DINOHead(embed_dim, 65536, False),
#             # )

#             dino.dino_utils.load_pretrained_weights(model, path, "teacher", arch, patch_size)

#         else:
#             assert(False)

#         print("PRETRAIN MODE", pretrain_mode)

#         st() # continue if loaded correctly

        

#         for p in model.parameters():
#             p.requires_grad = False
#         model.eval()
#         self.model = model

#         self.transform = transforms.Compose([
#             # transforms.Resize((480, 480)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])


#     def forward(self, x):
#         """Forward pass.
#         Args:
#             x (PIL): input data (image)
#         Returns:
#             tensor: depth
#         """

#         x = self.transform(x).unsqueeze(0)
#         attentions = self.model.get_last_selfattention(x.to(self.device))
#         # y = self.model.get_intermediate_layers(x.to(self.device), n=1)
#         # y = self.model(x.to(self.device)) #).get_intermediate_layers(x.to(self.device), n=1)
#         output = self.model.get_intermediate_layers(x.to(self.device), n=12)
#         out0 = output[0][:, 0]
#         out1 = output[1][:, 0]
#         out2 = output[2][:, 0]
#         out3 = output[3][:, 0]
#         out4 = output[4][:, 0]
#         out5 = output[5][:, 0]
#         out6 = output[6][:, 0]
#         out7 = output[7][:, 0]
#         out8 = output[8][:, 0]
#         out9 = output[9][:, 0]
#         out10 = output[10][:, 0]
#         out11 = output[11][:, 0]

#         # self.model(x.to(self.device))

#         out_dict = {}
#         out_dict["out0"] = out0
#         out_dict["out1"] = out1
#         out_dict["out2"] = out2
#         out_dict["out3"] = out3
#         out_dict["out4"] = out4
#         out_dict["out5"] = out5
#         out_dict["out6"] = out6
#         out_dict["out7"] = out7
#         out_dict["out8"] = out8
#         out_dict["out9"] = out9
#         out_dict["out10"] = out10
#         out_dict["out11"] = out11

#         return out_dict, attentions

class MIDAS(nn.Module):
    def __init__(self):
        super(MIDAS, self).__init__()

        # # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        # model_type = "MiDaS" # large model

        # self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=pretrained).eval()

        # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # self.transform = midas_transforms.default_transform

        # model_type = "MiDaS" # large model
        model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        self.midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True).eval()

        # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # self.transform = midas_transforms.default_transform

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def forward_vit(self, pretrained, x):
        b, c, h, w = x.shape

        glob = self.midas.pretrained.model.forward_flex(x)

        layer_1 = self.midas.pretrained.activations["1"]
        layer_2 = self.midas.pretrained.activations["2"]
        layer_3 = self.midas.pretrained.activations["3"]
        layer_4 = self.midas.pretrained.activations["4"]

        layer_1 = self.midas.pretrained.act_postprocess1[0:2](layer_1)
        layer_2 = self.midas.pretrained.act_postprocess2[0:2](layer_2)
        layer_3 = self.midas.pretrained.act_postprocess3[0:2](layer_3)
        layer_4 = self.midas.pretrained.act_postprocess4[0:2](layer_4)

        unflatten = nn.Sequential(
            nn.Unflatten(
                2,
                torch.Size(
                    [
                        h // self.midas.pretrained.model.patch_size[1],
                        w // self.midas.pretrained.model.patch_size[0],
                    ]
                ),
            )
        )

        if layer_1.ndim == 3:
            layer_1 = unflatten(layer_1)
        if layer_2.ndim == 3:
            layer_2 = unflatten(layer_2)
        if layer_3.ndim == 3:
            layer_3 = unflatten(layer_3)
        if layer_4.ndim == 3:
            layer_4 = unflatten(layer_4)

        layer_1 = self.midas.pretrained.act_postprocess1[3 : len(self.midas.pretrained.act_postprocess1)](layer_1)
        layer_2 = self.midas.pretrained.act_postprocess2[3 : len(self.midas.pretrained.act_postprocess2)](layer_2)
        layer_3 = self.midas.pretrained.act_postprocess3[3 : len(self.midas.pretrained.act_postprocess3)](layer_3)
        layer_4 = self.midas.pretrained.act_postprocess4[3 : len(self.midas.pretrained.act_postprocess4)](layer_4)

        return layer_1, layer_2, layer_3, layer_4


    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """

        x = x * 255

        x = x.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.int32)

        # print(x.min(), x.max())

        x = self.transform(x).cuda()

        if self.midas.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = self.forward_vit(self.midas.pretrained, x)

        layer_1_rn = self.midas.scratch.layer1_rn(layer_1)
        layer_2_rn = self.midas.scratch.layer2_rn(layer_2)
        layer_3_rn = self.midas.scratch.layer3_rn(layer_3)
        layer_4_rn = self.midas.scratch.layer4_rn(layer_4)

        path_4 = self.midas.scratch.refinenet4(layer_4_rn)
        path_3 = self.midas.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.midas.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.midas.scratch.refinenet1(path_2, layer_1_rn)

        out = self.midas.scratch.output_conv(path_1)

        # plt.figure()
        # plt.imshow(out.squeeze().cpu().numpy())
        # plt.savefig('images/test.png')
        # st()

        # layer_1 = self.midas.pretrained.layer1(x)
        # layer_2 = self.midas.pretrained.layer2(layer_1)
        # layer_3 = self.midas.pretrained.layer3(layer_2)
        # layer_4 = self.midas.pretrained.layer4(layer_3)

        # layer_1_rn = self.midas.scratch.layer1_rn(layer_1)
        # layer_2_rn = self.midas.scratch.layer2_rn(layer_2)
        # layer_3_rn = self.midas.scratch.layer3_rn(layer_3)
        # layer_4_rn = self.midas.scratch.layer4_rn(layer_4)

        # path_4 = self.midas.scratch.refinenet4(layer_4_rn)
        # path_3 = self.midas.scratch.refinenet3(path_4, layer_3_rn)
        # path_2 = self.midas.scratch.refinenet2(path_3, layer_2_rn)
        # path_1 = self.midas.scratch.refinenet1(path_2, layer_1_rn)

        # out1 = torch.nn.Sequential(*(list(self.midas.scratch.output_conv)[0:1]))(path_1)
        # out2 = torch.nn.Sequential(*(list(self.midas.scratch.output_conv)[1:4]))(out1)
        # out3 = torch.nn.Sequential(*(list(self.midas.scratch.output_conv)[4:]))(out2)

        out_dict = {}
        out_dict["layer_2"] = layer_2
        out_dict["layer_4"] = layer_4
        out_dict["layer_2_rn"] = layer_2_rn
        out_dict["layer_4_rn"] = layer_4_rn
        out_dict["path_3"] = path_3
        out_dict["path_1"] = path_1
        out_dict["out"] = out
        # out_dict["out2"] = out2
        # out_dict["out3"] = out3

        return out_dict

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
        self.midas = MIDAS() #torch.nn.Sequential(*(list(midas.pretrained.children())+list(midas.scratch.children()))[:-1])
        self.midas.cuda()
        self.midas.eval()

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

        self.B = 1
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

            # print(len(self.dataloader), feature_size)
            # continue

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

                # if only_process_stim_ids:
                #     if file_ids not in stim_list: 
                #         continue

                # print('Images processed: ', idx)

                # self.summ_writer = utils_improc.Summ_writer(
                #     writer=self.writer,
                #     global_step=idx,
                #     set_name=set_name,
                #     log_freq=log_freq,
                #     fps=8,
                #     # just_gif=True,
                # )

                # assert(images.shape[2]==424)
                # assert(images.shape[3]==424)
                assert(images.shape[0]==1)
                # images = images.cpu().numpy() #squeeze(0)
                # images_pil = self.toPIL(images).convert('RGB')

                # if self.summ_writer.save_this:
                #     rgb_camX = images.cuda().float().unsqueeze(0)
                #     rgb_camX_norm = rgb_camX - 0.5
                #     self.summ_writer.summ_rgb('inputs/rgb', rgb_camX_norm)
                #     del rgb_camX, rgb_camX_norm

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
                with torch.no_grad():
                    out_dict = self.midas(images)

                if save_feats:
                    for layer in layers:
                        # feats_all[layer][idx] = out_dict[layer].squeeze().detach().cpu().numpy() 

                        # layer_ind = layer_map[layer]
                        feat_memX = out_dict[layer]

                        num_spatial = len(feat_memX.shape[2:])
                        feat_size_flat = np.prod(torch.tensor(feat_memX.shape[1:]).numpy())
                        
                        
                        # 3d pool
                        feat_size_thresh = 200000

                        while feat_size_flat>feat_size_thresh:
                            if num_spatial==3:
                                feat_memX = self.pool3d(feat_memX)
                            elif num_spatial==2:
                                feat_memX = self.pool2d(feat_memX)
                            else:
                                assert(False)
                            feat_size_flat = np.prod(torch.tensor(feat_memX.shape[1:]).numpy())

                        feat_memX = feat_memX.squeeze(0).cpu().numpy().astype(np.float32)
                        # print(feat_memX.shape)

                        if idx==0:
                            feature_size = tag_sizes[tag]
                            if num_spatial==3:
                                c,h,w,d = feat_memX.shape
                                feats_all[layer] = np.zeros((feature_size, c, h, w, d), dtype=np.float32)
                                # file_order = np.zeros(73000)
                            elif num_spatial==2:
                                c,h,w = feat_memX.shape
                                feats_all[layer] = np.zeros((feature_size, c, h, w), dtype=np.float32)
                                # file_order = np.zeros(73000)
                            else:
                                assert(False)

                        feats_all[layer][idx] = feat_memX

                

                # if save_feats:
                #     print(idx)
                #     for layer in layers:
                #         feats_all[layer][idx] = out_dict[layer].squeeze().detach().cpu().numpy() 

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