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
from nets.featnet import FeatNet

from tqdm import tqdm


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

import matplotlib.pyplot as plt
from matplotlib import cm
# import seaborn as sn

from pycocotools.coco import COCO

from backend import saverloader, inputs


import sys
# sys.path.append("XTConsistency")
# from modules.unet import UNet, UNetReshade

np.set_printoptions(precision=2)
np.random.seed(0)

import ipdb
st = ipdb.set_trace

do_feat3d = True

XTC_init = '/home/gsarch/repo/pytorch_disco/saved_checkpoints/XTC_checkpoints/rgb2depth_consistency_wimagenet.pth'
coco_images_path = '/lab_data/tarrlab/common/datasets/NSD_images'

set_name = 'midas00' # take output of entire network
set_name = 'midas_bottleneck' # take output of refinenet1 (path_1)
set_name = 'midas_all_layers' # take output of refinenet1 (path_1)
# set_name = 'midas_all_layers_init' # take output of refinenet1 (path_1)
set_name = 'midas_coco_DPT_Large'

layers = [
    # "layer_2",
    # "layer_4",
    "layer_2_rn",
    "layer_4_rn",
    # "path_3",
    # "path_1",
    # "out",
]

print("LAYERS", layers)

checkpoint_dir='checkpoints/' + set_name
log_dir='logs_grnn_coco'

do_viewnet = False
save_feats = True
plot_classes = False
only_process_stim_ids = False
do_dim_red = True
log_freq = 1000 # frequency for logging tensorboard
subj = 1 # subject number - supports: 1, 2, 7 - only for when only_process_stim_ids=True
pool_len = 2 # avg pool length
num_maxpool = 1 # number of max pools

pretrained = True

if only_process_stim_ids:
    stim_list = np.load(
        "/user_data/yuanw3/project_outputs/NSD/output/coco_ID_of_repeats_subj%02d.npy" % (subj)
    )
    stim_list = list(stim_list)

hyp.view_depth = 32
hyp.feat_init = '02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_train_ns_grnn00'
if do_viewnet:
    hyp.view_init = '02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_train_ns_grnn00'

output_dir = f'/lab_data/tarrlab/gsarch/encoding_model/{set_name}'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, coco_images_path, transform):
        super(ImageFolderWithPaths, self).__init__(coco_images_path, transform=transform)
        if only_process_stim_ids:
            image_ids = [int(self.imgs[i][0].split('/')[-1].split('.')[0]) for i in range(len(self.imgs))]
            idxes = [image_ids.index(cid) for cid in stim_list]
            self.imgs = [self.imgs[idx] for idx in idxes]

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

# class MIDAS(nn.Module):
#     def __init__(self):
#         super(MIDAS, self).__init__()

#         # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#         #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#         #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
#         model_type = "MiDaS" # large model

#         self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=pretrained).eval()


#     def forward(self, x):
#         """Forward pass.
#         Args:
#             x (tensor): input data (image)
#         Returns:
#             tensor: depth
#         """

#         layer_1 = self.midas.pretrained.layer1(x)
#         layer_2 = self.midas.pretrained.layer2(layer_1)
#         layer_3 = self.midas.pretrained.layer3(layer_2)
#         layer_4 = self.midas.pretrained.layer4(layer_3)

#         layer_1_rn = self.midas.scratch.layer1_rn(layer_1)
#         layer_2_rn = self.midas.scratch.layer2_rn(layer_2)
#         layer_3_rn = self.midas.scratch.layer3_rn(layer_3)
#         layer_4_rn = self.midas.scratch.layer4_rn(layer_4)

#         path_4 = self.midas.scratch.refinenet4(layer_4_rn)
#         path_3 = self.midas.scratch.refinenet3(path_4, layer_3_rn)
#         path_2 = self.midas.scratch.refinenet2(path_3, layer_2_rn)
#         path_1 = self.midas.scratch.refinenet1(path_2, layer_1_rn)

#         out1 = torch.nn.Sequential(*(list(self.midas.scratch.output_conv)[0:1]))(path_1)
#         out2 = torch.nn.Sequential(*(list(self.midas.scratch.output_conv)[1:4]))(out1)
#         out3 = torch.nn.Sequential(*(list(self.midas.scratch.output_conv)[4:]))(out2)

#         return out3, out2, out1, path_1, path_3, layer_4_rn, layer_2_rn, layer_4, layer_2


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

        # target_tasks = ['normal','depth','reshading'] #options for XTC model
        # #initialize XTC model
        # ### DEPTH MODEL %%%%%%%%%%
        # task_index = target_tasks.index('depth')
        # models = [UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5)]
        # self.XTCmodel_depth = models[task_index].cuda().eval()

        # # pretrained_model = 'consistency_wimagenet'
        # # path = os.path.join(XTC_loc, 'models', 'rgb2'+'depth'+'_'+pretrained_model+'.pth')
        # map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
        # model_state_dict = torch.load(XTC_init, map_location=map_location)
        # self.XTCmodel_depth.load_state_dict(model_state_dict)

        # transform = transforms.Compose([transforms.Resize(self.W),
        #                     transforms.CenterCrop(self.W),
        #                     transforms.ToTensor()])

        # midas depth estimation
        # midas = torch.hub.load("intel-isl/MiDaS", "MiDaS") #, _use_new_zipfile_serialization=True)
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.midas.cuda()
        # self.midas.eval()

        # midas but remove last layer
        self.midas = MIDAS() #torch.nn.Sequential(*(list(midas.pretrained.children())+list(midas.scratch.children()))[:-1])
        self.midas.cuda()
        for p in self.midas.parameters():
            p.requires_grad = False
        self.midas.eval()
        print(self.midas)

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.default_transform

        # self.W = 256
        # self.H = 256
        self.W = 384
        self.H = 384
        self.fov = 60
        hfov = float(self.fov) * np.pi / 180.
        self.pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.pix_T_camX[0,2] = self.W/2. 
        self.pix_T_camX[1,2] = self.H/2. 
        

        self.B = 1
        assert(self.B==1) 
        self.pix_T_camX = torch.from_numpy(self.pix_T_camX).cuda().unsqueeze(0).repeat(self.B,1,1).float()

        data_loader_transform = transforms.Compose([
                            transforms.ToTensor()])
        dataset = ImageFolderWithPaths(coco_images_path, transform=data_loader_transform) # our custom dataset
        
        # self.dataloader = torch.utils_DataLoader(dataset, batch_size=32, shuffle=False)
        # dataset = datasets.ImageFolder(coco_images_path, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.B, shuffle=False)

        # self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X

        # self.Z = 128
        # self.Y = 128
        # self.X = 128
        # bounds = torch.tensor([-12.0, 12.0, -12.0, 12.0, -12.0, 12.0]).cuda()
        # self.scene_centroid = torch.tensor([0.0, 0.0, 10.0]).unsqueeze(0).repeat(self.B,1).cuda()
        # self.vox_util = utils_vox.Vox_util(self.Z, self.Y, self.X, set_name, scene_centroid=self.scene_centroid, assert_cube=True, bounds=bounds)

        self.writer = SummaryWriter(log_dir + '/' + set_name, max_queue=10, flush_secs=1000)

        # self.avgpool3d = nn.AvgPool3d(2, stride=2)
        self.toPIL = transforms.ToPILImage()

        self.pool_len = 2
        self.pool3d = nn.AvgPool3d(self.pool_len, stride=self.pool_len)
        self.pool2d = nn.AvgPool2d(self.pool_len, stride=self.pool_len)

        # self.run_extract()
    
    def run_extract(self):

        # if do_dim_red:
        #     ch = 15 # 15 pcs well captures 95% of variance
        # else:
        #     ch = 256
        
        # if only_process_stim_ids:
        #     feats = np.zeros((10000, ch,192/2,192/2), dtype=np.float32)
        #     file_order = np.zeros(10000)
        # else:
        #     out3_feats = np.zeros((73000,1, 192, 192), dtype=np.float32)
        #     out2_feats= np.zeros((73000,4, 96, 96), dtype=np.float32)
        #     out1_feats= np.zeros((73000,4, 96, 96), dtype=np.float32)
        #     path_1_feats= np.zeros((73000,15, 48, 48), dtype=np.float32)
        #     path_3_feats= np.zeros((73000,20, 48, 48), dtype=np.float32)
        #     layer_4_rn_feats= np.zeros((73000,35, 12, 12), dtype=np.float32)
        #     layer_2_rn_feats= np.zeros((73000,120, 24, 24), dtype=np.float32)
        #     layer_4_feats= np.zeros((73000,100, 12, 12), dtype=np.float32)
        #     layer_2_feats= np.zeros((73000,200, 24, 24), dtype=np.float32)
        #     file_order = np.zeros(73000)

        feats_all = {}
        cat_names = []
        supcat_names = []
        cat_ids = []
        idx = 0
        for images, _, file_ids in tqdm(self.dataloader):

            # print('Images processed: ', idx)


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
                        feature_size = 73000 # tag_sizes[tag]
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

        if save_feats:
            for layer in layers:
                feats = feats_all[layer]
                # if block_average:
                #     print("Block averaging...")
                #     cur_feat_len = feats.shape[0]
                #     new_feat_len = cur_feat_len/block_average_len
                #     assert(new_feat_len.is_integer())
                #     new_feat_len = int(new_feat_len)
                #     indices_blocks = np.arange(0, cur_feat_len+1, block_average_len)
                #     new_dims = list(feats.shape[1:])
                #     new_dims = [new_feat_len] + new_dims
                #     feats_ = np.zeros(new_dims)
                #     for ii in range(new_feat_len):
                #         feats_[ii] = np.mean(feats[indices_blocks[ii]:indices_blocks[ii+1]], axis=0)
                #     if zscore_after_block_averaging:
                #         feats_ = scipy.stats.zscore(feats_, axis=0)
                #     feats = feats_
                #     print("new feature length is:", feats.shape)
                np.save(f'{output_dir}/{layer}.npy', feats)


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

            # rgb_camX = images.cuda().float()
            # # rgb_camX_norm = rgb_camX - 0.5
            # # self.summ_writer.summ_rgb('inputs/rgb', rgb_camX_norm)

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
            
            # # estimate depth
            # rgb_camX = (rgb_camX.permute(0,2,3,1).detach().cpu().numpy() * 255).astype(np.uint8)
            # input_batch = []
            # for b in range(self.B):
            #     input_batch.append(self.transform(rgb_camX[b]).cuda())
            # input_batch = torch.cat(input_batch, dim=0)
            # out3, out2, out1, path_1, path_3, layer_4_rn, layer_2_rn, layer_4, layer_2 = self.midas(input_batch)

            # layers_list = {'out3':out3,'out2':out2, 'out1':out1, 'path_1':path_1, 'path_3':path_3, 'layer_4_rn':layer_4_rn, 'layer_2_rn':layer_2_rn, 'layer_4':layer_4, 'layer_2':layer_2}
            
            # # depth_cam = (torch.max(depth_cam) - depth_cam) / 100.0

            # # depth_cam = self.XTCmodel_depth(rgb_camXs)

            # # # get depths in 0-100 range approx.
            # # depth_cam = (depth_cam / 0.7) * 100
            
            # # self.summ_writer.summ_depth('inputs/depth_map', feat_memX[0].squeeze().detach().cpu().numpy())

            # if save_feats:
            #     # pooling

            #     # out3 = self.pool2d(out3)
            #     # out2 = self.pool2d(out2)
            #     # out1 = self.pool2d(out1)
            #     # path_1 = self.pool2d(path_1)

            #     do_pool = ['out3', 'out2', 'out1', 'path_1', 'layer_2', 'layer_2_rn']
            #     do_pool2 = ['out2', 'path_1']
            #     do_reduce_chans = {'out2':4, 'out1':4, 'path_1':15, 'path_3':20, 'layer_4_rn':35, 'layer_2_rn':120, 'layer_4':100, 'layer_2':200}

            #     # layers_list = {'out2':out2, 'out1':out1, 'path_1':path_1, 'path_3':path_3, 'layer_4_rn':layer_4_rn, 'layer_2_rn':layer_2_rn, 'layer_4':layer_4, 'layer_2':layer_2}

            #     for key in list(layers_list.keys()):
            #         if key in do_pool:
            #             layers_list[key] = self.pool2d(layers_list[key])
            #         if key in do_pool2: 
            #             layers_list[key] = self.pool2d(layers_list[key])

            #         if key in list(do_reduce_chans.keys()):
            #             ch = do_reduce_chans[key]
            #             feat_memX = layers_list[key].squeeze()
            #             feat_memX = feat_memX.permute(0,2,3,1)
            #             b,h,w,c = feat_memX.shape
            #             feat_memX = feat_memX.reshape(b, h*w, c).cpu().numpy()
            #             feat_memX_red = np.zeros((b,ch,h,w), dtype=np.float32)
            #             for b_i in range(self.B):
            #                 feat_memX_ = feat_memX[b_i]
            #                 pca = PCA(n_components=ch)
            #                 feat_memX_ = pca.fit_transform(feat_memX_)
            #                 feat_memX_= feat_memX_.reshape(h,w,ch).transpose((2,0,1))
            #                 feat_memX_red[b_i] = feat_memX_
            #                 # print(key, feat_memX_.shape)
            #                 # print(key, np.cumsum(pca.explained_variance_ratio_)[-1])
            #             # feat_memX = feat_memX_red
            #             layers_list[key] = feat_memX_red
            #             del feat_memX_red
            #         if torch.is_tensor(layers_list[key]):
            #             layers_list[key] = layers_list[key].cpu().numpy()
            #         # print(key, layers_list[key].reshape(self.B, -1).shape)
            #         # print(key, layers_list[key].shape)
            #             # feat_memX = torch.from_numpy(feat_memX).cuda()

            #             # plt.plot(np.cumsum(pca.explained_variance_ratio_))
            #             # plt.xlabel('number of components')
            #             # plt.ylabel('cumulative explained variance')
            #             # plt.yticks(np.arange(0.0,1.0,0.05))
            #             # plt.savefig('images/variance_explained.png')
            #             # st()                
    
                # 2d pool
                # if self.num_maxpool>0:
                #     for nm in range(self.num_maxpool):
                #         feat_memX = self.pool2d(feat_memX)

                # feats.append(feat_memX.detach().cpu().numpy())
                # file_order.append(file_ids.detach().cpu().numpy())
                # st()
                # feat_memX = feat_memX.reshape(self.B, -1)
        #         out3_feats[idx:idx+self.B] = layers_list['out3'].astype(np.float32)
        #         out2_feats[idx:idx+self.B]= layers_list['out2'].astype(np.float32)
        #         out1_feats[idx:idx+self.B]= layers_list['out1'].astype(np.float32)
        #         path_1_feats[idx:idx+self.B]= layers_list['path_1'].astype(np.float32)
        #         path_3_feats[idx:idx+self.B]= layers_list['path_3'].astype(np.float32)
        #         layer_4_rn_feats[idx:idx+self.B]= layers_list['layer_4_rn'].astype(np.float32)
        #         layer_2_rn_feats[idx:idx+self.B]= layers_list['layer_2_rn'].astype(np.float32)
        #         layer_4_feats[idx:idx+self.B]= layers_list['layer_4'].astype(np.float32)
        #         layer_2_feats[idx:idx+self.B]= layers_list['layer_2'].astype(np.float32)
        #         file_order[idx:idx+self.B] = file_ids.detach().cpu().numpy()                

        #     idx += 1*self.B

        #     # plt.close('all')

        #     # if idx == 10:
        #     #     break

        #     if only_process_stim_ids:
        #         if idx == 10000:
        #             break
        
        # # feats = np.concatenate(feats, axis=0)
        # # file_order = np.concatenate(file_order, axis=0)
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

        # if save_feats:
        #     np.save(f'{output_dir}/out3_feats.npy', out3_feats)
        #     np.save(f'{output_dir}/out2_feats.npy', out2_feats)
        #     np.save(f'{output_dir}/out1_feats.npy', out1_feats)
        #     np.save(f'{output_dir}/path_1_feats.npy', path_1_feats)
        #     np.save(f'{output_dir}/path_3_feats.npy', path_3_feats)
        #     np.save(f'{output_dir}/layer_4_rn_feats.npy', layer_4_rn_feats)
        #     np.save(f'{output_dir}/layer_2_rn_feats.npy', layer_2_rn_feats)
        #     np.save(f'{output_dir}/layer_4_feats.npy', layer_4_feats)
        #     np.save(f'{output_dir}/layer_2_feats.npy', layer_2_feats)
        #     np.save(f'{output_dir}/file_order.npy', file_order)
        #     if plot_classes:
        #         np.save(f'{output_dir}/supcat.npy', np.array(supcat_names))
        #         np.save(f'{output_dir}/cat.npy', np.array(cat_names))

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

    def plot_tsne_only(self):

        self.summ_writer = utils_improc.Summ_writer(
                writer=self.writer,
                global_step=1,
                log_freq=1,
                fps=8,
                just_gif=True,
            )

        feats = np.load(f'{output_dir}/replica_carla_feats.npy')
        # file_order = np.load(f'{output_dir}/replica_carla_file_order.npy')
        supcat_names = np.load(f'{output_dir}/replica_carla_supcat.npy')
        cat_names = np.load(f'{output_dir}/replica_carla_cat.npy')

        tsne = TSNE(n_components=2).fit_transform(feats)
        # pred_catnames_feats = [self.maskrcnn_to_ithor[i] for i in self.feature_obj_ids]
            

        # # tsne plot colored by predicted labels
        # tsne_pred_figure = self.get_colored_tsne_image(supcat_names, tsne)
        # self.summ_writer.summ_figure(f'tsne/tsne_grnn_reduced_supcat', tsne_pred_figure)

        # tsne_pred_figure = self.get_colored_tsne_image(cat_names, tsne)
        # self.summ_writer.summ_figure(f'tsne/tsne_grnn_reduced_cat', tsne_pred_figure)

        self.plot_by_classes(supcat_names, tsne, self.summ_writer)
        
        

        # if as_image: 
        #     image = self.plot_to_image(figure)
        #     return image
        # else:
        #     return figure


    def plot_depth(self):

        self.summ_writer = utils_improc.Summ_writer(
                writer=self.writer,
                global_step=1,
                log_freq=1,
                fps=8,
                just_gif=True,
            )

        feats = np.load(f'{output_dir}/replica_carla_feats.npy')#[:10]
        # file_order = np.load(f'{output_dir}/replica_carla_file_order.npy')
        supcat_names = np.load(f'{output_dir}/replica_carla_supcat.npy')#[:10]
        cat_names = np.load(f'{output_dir}/replica_carla_cat.npy')#[:10]
        


        median = []
        mean = []
        idx = 0
        for images, _, file_ids in self.dataloader:

            print('Images processed: ', idx)

            self.summ_writer = utils_improc.Summ_writer(
                writer=self.writer,
                global_step=idx,
                log_freq=50,
                fps=8,
                just_gif=True,
            )

            rgb_camX = images.cuda().float()
            
            # estimate depth
            rgb_camX = (rgb_camX.permute(0,2,3,1).detach().cpu().numpy() * 255).astype(np.uint8)
            input_batch = []
            for b in range(self.B):
                input_batch.append(self.transform(rgb_camX[b]).cuda())
            input_batch = torch.cat(input_batch, dim=0)
            with torch.no_grad():
                depth_cam = self.midas(input_batch).unsqueeze(1)
                depth_cam = (torch.max(depth_cam) - depth_cam) / 100.0


            # for b in range(self.B):
            depth_cam = depth_cam - torch.min(depth_cam.reshape(self.B, -1), dim=1).values.reshape(-1, 1, 1, 1)
            depth_cam = depth_cam / torch.max(depth_cam.reshape(self.B, -1), dim=1).values.reshape(-1, 1, 1, 1)

            median.append(torch.median(depth_cam.reshape(self.B, -1), dim=1).values)
            mean.append(torch.mean(depth_cam.reshape(self.B, -1), dim=1))

            idx += 1*self.B

            # if idx == 100:
            #     break

        median = torch.cat(median, dim=0)
        mean = torch.cat(mean, dim=0)
        data_to_plot = torch.stack([median, mean]).t().detach().cpu().numpy()

        tx = data_to_plot[:, 0]
        ty = data_to_plot[:, 1]
        # tx = self.scale_to_01_range(tx)
        # ty = self.scale_to_01_range(ty)

        unique_classes = []
        unique_classes.extend(supcat_names)
        unique_classes = list(set(unique_classes))

        markers = ['^', 'X', 'o', 's', 'p']
        num_markers = len(markers)

        # predicted clusters
        evenly_spaced_interval = np.linspace(0, 1, len(unique_classes))
        colors = [cm.gist_rainbow(x) for x in evenly_spaced_interval]
        for idx in range(len(unique_classes)):
            figure = plt.figure(figsize = (20,20))
            ax = figure.add_subplot(111)

            label = unique_classes[idx]
            indices = [i for i, l in enumerate(supcat_names) if l == label]

            current_tx = np.take(tx, indices)
            # current_ty = np.take(ty, indices)

            marker = markers[idx%num_markers]
            color = colors[idx]
            ax.hist(current_tx, bins=np.arange(0,1,0.05))
            plt.xlabel('median')
            # ax.ylabel('mean')
            # ax.scatter(current_tx, current_ty, c=color, marker=marker, label=label)

            # ax.legend(loc='best')

            # if x_label is not None:
            #     plt.xlabel(x_label)
            # if y_label is not None:
            #     plt.ylabel(y_label)

            self.summ_writer.summ_figure(f'depth_supcat/depth_median_{label}', figure)

            plt.close()

        # # tsne plot colored by predicted labels
        # tsne_pred_figure = self.get_colored_tsne_image(supcat_names, data_to_plot)
        # plt.xlabel('median')
        # plt.ylabel('mean')
        # self.summ_writer.summ_figure(f'depth/depth_supcat', tsne_pred_figure)

        # tsne_pred_figure = self.get_colored_tsne_image(cat_names, data_to_plot)
        # plt.xlabel('median')
        # plt.ylabel('mean')
        # self.summ_writer.summ_figure(f'depth/depth_cat', tsne_pred_figure)

        # self.plot_by_classes(supcat_names, data_to_plot, self.summ_writer, x_label='median', y_label='mean')


    def plot_by_classes(self, catnames, data_to_plot, summ_writer, x_label=None, y_label=None):

        tx = data_to_plot[:, 0]
        ty = data_to_plot[:, 1]
        tx = self.scale_to_01_range(tx)
        ty = self.scale_to_01_range(ty)

        unique_classes = []
        unique_classes.extend(catnames)
        unique_classes = list(set(unique_classes))

        markers = ['^', 'X', 'o', 's', 'p']
        num_markers = len(markers)

        # predicted clusters
        evenly_spaced_interval = np.linspace(0, 1, len(unique_classes))
        colors = [cm.gist_rainbow(x) for x in evenly_spaced_interval]
        for idx in range(len(unique_classes)):
            figure = plt.figure(figsize = (20,20))
            ax = figure.add_subplot(111)

            label = unique_classes[idx]
            indices = [i for i, l in enumerate(catnames) if l == label]

            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            marker = markers[idx%num_markers]
            color = colors[idx]
            ax.scatter(current_tx, current_ty, c=color, marker=marker, label=label)

            ax.legend(loc='best')

            if x_label is not None:
                plt.xlabel(x_label)
            if y_label is not None:
                plt.ylabel(y_label)

            summ_writer.summ_figure(f'depth_classes/depth_supcat_{label}', figure)

            plt.close()


    def get_colored_tsne_image(self, catnames, tsne, as_image=False):
        tx = tsne[:, 0]
        ty = tsne[:, 1]
        tx = self.scale_to_01_range(tx)
        ty = self.scale_to_01_range(ty)

        figure = plt.figure(figsize = (20,20))
        ax = figure.add_subplot(111)

        unique_classes = []
        unique_classes.extend(catnames)
        unique_classes = list(set(unique_classes))

        markers = ['^', 'X', 'o', 's', 'p']
        num_markers = len(markers)

        # predicted clusters
        evenly_spaced_interval = np.linspace(0, 1, len(unique_classes))
        colors = [cm.gist_rainbow(x) for x in evenly_spaced_interval]
        for idx in range(len(unique_classes)):
            label = unique_classes[idx]
            indices = [i for i, l in enumerate(catnames) if l == label]

            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            marker = markers[idx%num_markers]
            color = colors[idx]
            ax.scatter(current_tx, current_ty, c=color, marker=marker, label=label)
        
        ax.legend(loc='best')

        if as_image: 
            image = self.plot_to_image(figure)
            return image
        else:
            return figure

    def scale_to_01_range(self,x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range


if __name__ == '__main__':
    model = CARLA_MOC(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    model.initialize_model()