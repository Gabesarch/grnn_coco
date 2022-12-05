import os
# script = """
MODE="CLEVR_STA"
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

from torchvision import datasets, models, transforms
import torchvision
# from torchvision import models as torchvision_models

import matplotlib.pyplot as plt
from matplotlib import cm
# import seaborn as sn

from pycocotools.coco import COCO
import sys

import pandas
from tqdm import tqdm

# sys.path.append("XTConsistency")
# from modules.unet import UNet, UNetReshade

np.random.seed(0)

import ipdb
st = ipdb.set_trace

from torch.utils import model_zoo

sys.path.append("/home/gsarch/repo/nsd/")
from util.util import dimensionality_reduction_feats
from sklearn.model_selection import train_test_split
from featureprep.feature_prep import extract_feature_with_image_order

sys.path.append("dino")
import dino.dino_utils
from vision_transformer import DINOHead

##########%%%%%%%%% PARAMETERS %%%%%%%%%%%%##################%%%%%%%%%%%%%%%%%########
model_load = 'resnet50_trained_on_SIN' # dino_replica_carla, simclr_replica_carla, resnet50_trained_on_IN, resnet50_trained_on_SIN, resnet50_trained_on_SIN_and_IN, resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN
model_load = 'vicregl_resnet50_alpha0p9'
model_load = 'vicregl_resnet50_alpha0p75'
only_process_stim_ids = False
save_full_features = False # save full features
save_subject_pca_features = True # save pca features for each subject (fit on training images, apply to testing images)
write_to_excel = False # append info to model_info_file
log_freq = 30000 # frequency for logging tensorboard
subjects = [1,2,3,4,5,6,7,8] # subjects to save pca for if save_subject_pca_features=True
batch_size = 50 # batch size for images
num_layers_per_batch = 1 # number of layers to process at a time
# layers = [
#     'out_initial',
#     'out_layer1',
#     'out_layer2',
#     'out_layer3',
#     'out_layer4',
#     'out_avgpool',
#     # 'fc1_out',
#     # 'fc2_out',
# ]
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
if model_load in ["simclr_replica_carla", "dino_replica_carla"]:
    layers = layers + ['out_fc1','out_fc2']
XTC_init = '/home/gsarch/repo/pytorch_disco/saved_checkpoints/XTC_checkpoints/rgb2depth_consistency_wimagenet.pth'
coco_images_path = '/lab_data/tarrlab/common/datasets/NSD_images'
stim_dir='/user_data/yuanw3/project_outputs/NSD/output'
model_info_file = '/home/gsarch/repo/nsd/model_info.xlsx'
general_model_name = 'resnet50'
dataset = 'nsd'
##########%%%%%%%%%%%%%%%%%%%%%##################%%%%%%%%%%%%%%%%%%%#####################
print("MODEL IS", model_load)

# layer_groups = list(zip(*[iter(layers)]*num_layers_per_batch))
# num_layers_ = sum([len(layer_groups_i) for layer_groups_i in layer_groups])
# if num_layers_ < len(layers):
#     layer_groups.append(layers[num_layers_:])
def group_list(list_to_group, num_group):
    layer_groups = list(zip(*[iter(list_to_group)]*num_group))
    num_layers_ = sum([len(layer_groups_i) for layer_groups_i in layer_groups])
    if num_layers_ < len(list_to_group):
        layer_groups.append(tuple(list_to_group[num_layers_:]))
    return layer_groups
layer_groups = group_list(layers, num_layers_per_batch)
print(layer_groups)
set_name = model_load
checkpoint_dir='checkpoints/' + set_name
log_dir='logs_resnet_coco'

output_dir = f'/lab_data/tarrlab/gsarch/encoding_model/{set_name}'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py

# model_info_table = pandas.read_excel(model_info_file)

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, coco_images_path, transform):
        super(ImageFolderWithPaths, self).__init__(coco_images_path, transform=transform)
        # if only_process_stim_ids:
        #     image_ids = [int(self.imgs[i][0].split('/')[-1].split('.')[0]) for i in range(len(self.imgs))]
        #     idxes = [image_ids.index(cid) for cid in stim_list]
        #     self.imgs = [self.imgs[idx] for idx in idxes]

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        path = path.split('/')[-1].split('.')[0]
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
        if model_load=='dino_replica_carla':
            path = '/user_data/gsarch/dino_output_renset50/checkpoint0300.pth'
            checkpoint = torch.load(path)
            key_ = "teacher"
            state_dict = checkpoint[key_]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            checkpoint["state_dict"] = state_dict
        elif model_load=='simclr_replica_carla':
            path = '/user_data/gsarch/simclr_output_300epoch/checkpoint_0300.pth.tar'
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
        elif model_load=='vicregl_resnet50_alpha0p9':
            model = torch.hub.load('facebookresearch/vicregl:main', 'resnet50_alpha0p9')
        elif model_load=='vicregl_resnet50_alpha0p75':
            model = torch.hub.load('facebookresearch/vicregl:main', 'resnet50_alpha0p75')
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
        elif 'vicregl' in model_load:
            pass
        else:
            model = models.resnet50(pretrained=pretrained)

            if not pretrained:
                model.load_state_dict(checkpoint["state_dict"])

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

class RUN_COCO_EXTRACT(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = COCO_EXTRACT()

        self.model.eval()

        # self.start_iter = saverloader.load_weights(self.model, None)

        self.model.run_extract()
        # self.model.plot_tsne_only()
        # self.model.plot_depth()

class COCO_EXTRACT(nn.Module):
    def __init__(self):
        super(COCO_EXTRACT, self).__init__()

        # path = '/home/gsarch/repo/3DQNets/pytorch_disco/checkpoints/02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_train_ns_grnn00/model-60000.pth'
        # checkpoint = torch.load(path)
        # self.feat3dnet.load_state_dict(checkpoint['model_state_dict'])
        if False:
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
        self.model = RESNET() #torch.nn.Sequential(*(list(midas.pretrained.children())+list(midas.scratch.children()))[:-1])
        self.model.cuda()
        self.model.eval()
        print(self.model)
        

        self.B = batch_size # batch size

        data_loader_transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])
        dataset = ImageFolderWithPaths(coco_images_path, transform=data_loader_transform) # our custom dataset
        
        # self.dataloader = torch.utils_DataLoader(dataset, batch_size=32, shuffle=False)
        # dataset = datasets.ImageFolder(coco_images_path, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.B, shuffle=False)

        self.writer = SummaryWriter(log_dir + '/' + set_name, max_queue=10, flush_secs=1000)

        self.toPIL = transforms.ToPILImage()

        self.pool_len = 2
        self.pool3d = nn.AvgPool3d(self.pool_len, stride=self.pool_len)
        self.pool2d = nn.AvgPool2d(self.pool_len, stride=self.pool_len)

    def run_extract(self):

        
        for layer_batch in tqdm(layer_groups):
            idx = 0
            file_order = np.zeros(73000)
            feats = {}
            if write_to_excel:
                model_info_table = pandas.read_excel(model_info_file)
                if 'Unnamed: 0' in model_info_table.keys():
                    model_info_table = model_info_table.drop('Unnamed: 0', 1)
            df_appends = []
            model_names_info_table_added = []
            for images, _, file_ids in tqdm(self.dataloader, leave=False):

                # if only_process_stim_ids:
                #     if file_ids not in stim_list: 
                #         continue

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
                # assert(images.shape[0]==1)
                # images = images.squeeze(0)
                # images_pil = images.cpu().numpy()
                # images_pil = [self.toPIL(image).convert('RGB') for image in images]

                if self.summ_writer.save_this:
                    rgb_camX = images[0].cuda().float().unsqueeze(0)
                    rgb_camX_norm = rgb_camX - 0.5
                    self.summ_writer.summ_rgb('inputs/rgb', rgb_camX_norm)
                    del rgb_camX, rgb_camX_norm

                # if False:
                #     plt.figure()
                #     rgb_camXs_np = rgb_camXs[0].permute(1,2,0).detach().cpu().numpy()
                #     plt.imshow(rgb_camXs_np)
                #     plt.savefig('images/test.png')


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
                
                # out0, layer1_out, layer2_out, layer3_out, layer4_out, avgpool_out = self.dino(images_pil)

                # out0 = self.avgpool2d(out0)
                # for n in range(3):
                #     layer1_out = self.avgpool2d(layer1_out)
                # for n in range(2):
                #     layer2_out = self.avgpool2d(layer2_out)
                # for n in range(2):
                #     layer3_out = self.avgpool2d(layer3_out)
                # layer4_out = self.avgpool2d(layer4_out)

                # if save_feats:
                #     out0_feats[idx] = out0.squeeze().detach().cpu().numpy() 
                #     layer1_out_feats[idx] = layer1_out.squeeze().detach().cpu().numpy() 
                #     layer2_out_feats[idx] = layer2_out.squeeze().detach().cpu().numpy() 
                #     layer3_out_feats[idx] = layer3_out.squeeze().detach().cpu().numpy() 
                #     layer4_out_feats[idx] = layer4_out.squeeze().detach().cpu().numpy()
                #     avgpool_out_feats[idx] = avgpool_out.squeeze().detach().cpu().numpy()
                #     file_order[idx] = file_ids.detach().cpu().numpy() 

                with torch.no_grad():
                    feats_all = self.model(images)

                # layer_ind = layer_map[layer]
                for layer_ in layer_batch:
                    feat_memX = feats_all[layer_]

                    num_spatial = len(feat_memX.shape[2:])
                    feat_size_flat = np.prod(torch.tensor(feat_memX.shape[1:]).numpy())
                    
                    # 3d pool
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
                        if num_spatial==3:
                            b,c,h,w,d = feat_memX.shape
                            feats[layer_] = np.zeros((73000, c, h, w, d), dtype=np.float32)
                        elif num_spatial==2:
                            b,c,h,w = feat_memX.shape
                            feats[layer_] = np.zeros((73000, c, h, w), dtype=np.float32)
                            # file_order = np.zeros(73000)
                        elif num_spatial==1:
                            b,c,h = feat_memX.shape
                            feats[layer_] = np.zeros((73000, c, h), dtype=np.float32)
                            # file_order = np.zeros(73000)
                        elif num_spatial==0:
                            b,c = feat_memX.shape
                            feats[layer_] = np.zeros((73000, c), dtype=np.float32)
                            # file_order = np.zeros(73000)
                        else:
                            assert(False)

                    feats[layer_][idx:idx+self.B] = feat_memX
                    file_order[idx:idx+self.B] = file_ids.detach().cpu().numpy()     

                idx += 1*self.B

                # if idx > 100:
                #     break

                # plt.close('all')

                # if idx == 10:
                #     assert(False)

                # if only_process_stim_ids:
                #     if idx == 10000:
                #         break
            
            if save_subject_pca_features:
                print("Saving features")

                for layer_ in layer_batch:

                    for subj in subjects:

                        print(f"Saving layer {layer_}, subject {subj}")

                        pca_train_file_path =  f"{output_dir}/{layer_}_subj{subj}_train_pca.npy"
                        pca_test_file_path =  f"{output_dir}/{layer_}_subj{subj}_test_pca.npy"
                        pca_fit_path = f"{output_dir}/{layer_}_subj{subj}_pca_fit.p" # path to pca model

                        if os.path.isfile(pca_train_file_path) and os.path.isfile(pca_test_file_path):
                            print(f"{pca_train_file_path} exists. delete old file to regenerate.")
                            continue

                        stimulus_list = np.load(
                            "%s/coco_ID_of_repeats_subj%02d.npy" % (stim_dir, subj)
                        )
                        feature_mat = extract_feature_with_image_order(
                            stimulus_list, feats[layer_], file_order, 
                        )

                        # split train and test set
                        X_train, X_test = train_test_split(
                            feature_mat, test_size=0.15, random_state=42
                        )

                        _,_ = dimensionality_reduction_feats(X_train, X_test, layer_, pca_train_file_path, pca_test_file_path, pca_fit_path, method='pca', override_existing_pca=True)

                        del feature_mat, X_train, X_test

                        if write_to_excel:
                            entry_name = f'{model_load}_{layer_}'
                            if (entry_name in model_info_table['model_name'].tolist()) or (entry_name in model_names_info_table_added):
                                print("entry already exists.. skipping")
                            else:
                                # last_index = model_info_table[model_info_table.keys()[0]].keys()[-1]
                                data_to_append = [[entry_name, general_model_name, f'{output_dir}/{layer_}.npy', f'{output_dir}/file_order.npy', dataset, 'all_pca', '', '']]
                                df_append = pandas.DataFrame(data_to_append, columns=list(model_info_table.keys()))
                                df_appends.append(df_append)
                                model_names_info_table_added.append(entry_name)
                                # model_info_table = model_info_table.append(df_append, ignore_index=True)

            if save_full_features:
                
                for layer_ in layer_batch:
                    print("Saving full features ", layer_)
                    np.save(f'{output_dir}/{layer_}.npy', feats[layer_])

                    if write_to_excel:
                        entry_name = f'{model_load}_{layer_}'
                        if (entry_name in model_info_table['model_name'].tolist()) or (entry_name in model_names_info_table_added):
                            print("entry already exists.. skipping")
                        else:
                            # last_index = model_info_table[model_info_table.keys()[0]].keys()[-1]
                            data_to_append = [[entry_name, general_model_name, f'{output_dir}/{layer_}.npy', f'{output_dir}/file_order.npy', dataset, 'all_pca', '', '']]
                            df_append = pandas.DataFrame(data_to_append, columns=['model_name', 'model', 'feature_path', 'feature_order_path', 'dataset', 'subjects', 'size', 'description'])
                            df_appends.append(df_append)
                            model_names_info_table_added.append(entry_name)
                            # with pandas.ExcelWriter(model_info_file, mode='a', engine="openpyxl", if_sheet_exists='overlay') as writer:
                            #     df_append.to_excel(writer, sheet_name='Sheet1')
                            # model_info_table = model_info_table.append(df_append, ignore_index=True)

            np.save(f'{output_dir}/file_order.npy', file_order)

            if write_to_excel:
                # model_info_table = model_info_table.drop('Unnamed: 0.1', 1)
                # model_info_table = model_info_table.drop('Unnamed: 0', 1)
                # model_info_table.reset_index(drop=True, inplace=True)
                # model_info_table = model_info_table.drop(np.arange(180, 182), 0)
                model_info_table = pandas.read_excel(model_info_file)
                if 'Unnamed: 0' in model_info_table.keys():
                    model_info_table = model_info_table.drop('Unnamed: 0', 1)
                for df_append in df_appends:
                    model_info_table = model_info_table.append(df_append, ignore_index=True)
                with pandas.ExcelWriter(model_info_file, engine="openpyxl") as writer:
                    model_info_table.to_excel(writer)
                # writer = pandas.ExcelWriter(model_info_file)
                # model_info_table.to_excel(writer)
                # writer.save()
                # writer.close()
                # writer.handles = None
                # with pandas.ExcelWriter(model_info_file, mode='a') as writer:
                #     df.to_excel(writer, sheet_name='Sheet3')
                print('DataFrame is written successfully to Excel File.')

if __name__ == '__main__':
    model = RUN_COCO_EXTRACT(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    model.initialize_model()