import os
MODE="CARLA_MOC"
os.environ["MODE"] = MODE
os.environ["exp_name"] = 'trainer_carla_replica_feat3d_enc3d_occ_view_vox'
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
from nets.feat3dnet import Feat3dNet
# from nets.emb3dnet import Emb3dNet
from nets.occnet import OccNet
from nets.viewnet import ViewNet

import sys
sys.path.append("/home/gsarch/repo/nsd/")
from util.util import dimensionality_reduction_feats
from sklearn.model_selection import train_test_split
from featureprep.feature_prep import extract_feature_with_image_order


import cv2

from tensorboardX import SummaryWriter
import torch.nn.functional as F
import utils.samp
import utils.geom
import utils.improc
import utils.basic
import utils.eval
import utils.py
import utils.misc
import utils.track
import utils.vox

from torchvision import transforms
from torchvision import datasets

import matplotlib.pyplot as plt
from matplotlib import cm
# import seaborn as sn

from pycocotools.coco import COCO

from backend import saverloader, inputs

import ipdb
st = ipdb.set_trace
coco_images_path = '/lab_data/tarrlab/common/datasets/NSD_images'

layers = [
    'occ_e',
    'x0',
    'x1',
    'x2',
    'x3',
    'x4',
    'x5',
    'x6',
    'x7',
    'x8',
    'x9',
    'feat_norm',
]

set_name = 'enc3d_emb3d_07'
set_name = 'enc3d_emb3d_omnidatafull_bounds_00'
set_name = 'enc3d_emb3d_omnidatafull_nobounds_00'
# set_name = 'GRNN_test02'
# set_name = 'omnidata_test'
set_name = 'enc3d_emb3d_omnidatafull_bounds_01'
set_name = 'enc3d_emb3d_omnidatafull_bounds_metricce'
set_name = 'enc3d_emb3d_omnidatafull_nobounds_metricce_rotaterand_00'
set_name = 'test_00_extract_coco'
set_name = 'enc3d_emb3d_omnidatafull_nobounds_metricce_rotaterand_fix00'
set_name = 'enc3d_emb3d_omnidatafull_nobounds_metricce_rotaterand_fix_nopooling00'




print(set_name)

checkpoint_dir='checkpoints/' + set_name
log_dir='logs_omnidata_moc'

# do_viewnet = True
flip_layer_order = True
save_full_features = False
save_subject_pca_features = True
plot_classes = False
only_process_stim_ids = False
reduce_channels = False
log_freq = 500 # frequency for logging tensorboard
# log_freq = 1 # frequency for logging tensorboard
subjects = [1,2,3,4,5,6,7,8] # subjects to save pca for if save_subject_pca_features=True
pretrain = True
# print("SUBJECT:", subj)

stim_dir='/user_data/yuanw3/project_outputs/NSD/output'

if flip_layer_order:
    layers = list(np.flip(np.array(layers)))

if only_process_stim_ids:
    stim_list = np.load(
        "/user_data/yuanw3/project_outputs/NSD/output/coco_ID_of_repeats_subj%02d.npy" % (subj)
    )
    stim_list = list(stim_list)

# hyp.view_depth = 32
# hyp.feat_init = '02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_train_ns_grnn00'
# hyp.feat3d_init = '01_s2_m128x128x128_1e-4_F3_d32_O_c1_s.1_E3_n2_d16_c1_carla_and_replica_train_cr12'
hyp.feat3d_init = '01_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_ns_enc3d_emb3d_07_fitvox'
# hyp.view_init = '01_m144x144x144_p128x384_1e-4_O_c1_s.1_V_d32_c1_carla_and_replica_train_ns_enc3d_view07_fitvox'
hyp.occ_init = '01_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_ns_enc3d_emb3d_07_fitvox'
# if do_viewnet:
#     hyp.view_init = '02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_train_ns_grnn00'

hyp.feat3d_init = '02_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_carla_and_replica_val_ns_moc_omnidata_bounds_00'
hyp.occ_init = '02_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_carla_and_replica_val_ns_moc_omnidata_bounds_00'

hyp.feat3d_init = '02_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_carla_and_replica_val_ns_moc_omnidata_nobounds_01'
hyp.occ_init = '02_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_carla_and_replica_val_ns_moc_omnidata_nobounds_01'

hyp.feat3d_init = '02_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_carla_and_replica_val_ns_moc_omnidata_bounds_02'
hyp.occ_init = '02_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_carla_and_replica_val_ns_moc_omnidata_bounds_02'


hyp.feat3d_init = '02_m144x144x144_1e-4_O_c1_carla_and_replica_train_carla_and_replica_val_ns_moc_omnidata_bounds_tiny_00'
hyp.occ_init = '02_m144x144x144_1e-4_O_c1_carla_and_replica_train_carla_and_replica_val_ns_moc_omnidata_bounds_tiny_00'

hyp.feat3d_init = '02_m144x144x144_1e-4_O_c1_carla_and_replica_train_carla_and_replica_val_ns_moc_omnidata_tiny_nobounds_rotaterand_00'
hyp.occ_init = '02_m144x144x144_1e-4_O_c1_carla_and_replica_train_carla_and_replica_val_ns_moc_omnidata_tiny_nobounds_rotaterand_00'

hyp.feat3d_init = '04_m144x36x144_1e-4_O_c1_s1_carla_and_replica_train_carla_and_replica_val_ns_omnidata_moc_boundsadjusted_occ_emb3d_pretrainocc_00'
hyp.occ_init = '04_m144x36x144_1e-4_O_c1_s1_carla_and_replica_train_carla_and_replica_val_ns_omnidata_moc_boundsadjusted_occ_emb3d_pretrainocc_00'

Z, Y, X = 144, 36, 144
# output_dir = f'/lab_data/tarrlab/gsarch/encoding_model/{set_name}_subj={subj}_pl={pool_len}_nm={num_maxpool}'
# output_dir = '/lab_data/tarrlab/gsarch/encoding_model/grnn_feats_init_32x18x18x18' #/subj%s' % (subj)
# output_dir = f'/lab_data/tarrlab/gsarch/encoding_model/{set_name}'
# output_dir = f'/user_data/gsarch/encoding_model/{set_name}'
output_dir = f'/lab_data/tarrlab/scratch/gsarch/encoding_model/{set_name}'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
print("output_dir: ", output_dir)

class GRNN(nn.Module):
    def __init__(self):
        super(GRNN, self).__init__()

        feat3dnet = Feat3dNet(in_dim=4).cuda()

        occnet = OccNet()

        viewnet = ViewNet(feat_dim=hyp.feat3d_dim)

        if pretrain:
            inits = {"feat3dnet": hyp.feat3d_init,
                #  "viewnet": hyp.view_init,
                 "occnet": hyp.occ_init}
            # print(inits)
            for part, init in list(inits.items()):
                # st()
                if init:
                    if part == 'feat3dnet':
                        model_part = feat3dnet
                    elif part == 'occnet':
                        model_part = occnet
                    elif part == 'viewnet':
                        model_part = viewnet
                    else:
                        assert(False)
                    iter = saverloader.load_part(model_part, part, init)
                    if iter:
                        print("loaded %s at iter %d" % (init, iter))
                    else:
                        print("could not find a checkpoint for %s" % init)
        
        # for p in featnet.parameters():
        #     p.requires_grad = False
        # featnet.eval()

        # feat3dnet encoder
        self.encoder_layer0 = feat3dnet.net.encoder_layer0
        self.encoder_layer1 = feat3dnet.net.encoder_layer1
        self.encoder_layer2 = feat3dnet.net.encoder_layer2
        self.encoder_layer3 = feat3dnet.net.encoder_layer3
        self.encoder_layer4 = feat3dnet.net.encoder_layer4
        self.encoder_layer5 = feat3dnet.net.encoder_layer5
        self.encoder_layer6 = feat3dnet.net.encoder_layer6
        self.encoder_layer7 = feat3dnet.net.encoder_layer7
        self.encoder_layer8 = feat3dnet.net.encoder_layer8
        self.final_layer = feat3dnet.net.final_layer
        
        # occnet 
        self.occ_conv = occnet.conv3d

        self.W, self.H = 256, 256
        self.fov = 90
        self.hfov = float(self.fov) * np.pi / 180.
        self.pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(self.hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(self.hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.pix_T_camX[0,2] = self.W/2. 
        self.pix_T_camX[1,2] = self.H/2.         

        self.B = 1 
        # assert(self.B==1) # need B=1 for this - TODO: allow batching
        self.pix_T_camX = torch.from_numpy(self.pix_T_camX).cuda().unsqueeze(0).repeat(self.B,1,1).float()

    
    def forward(self, feat_input, vox_util, pix_T_camX, summ_writer=None, norm=True):

        # self.pix_T_camX = pix_T_camX

        B, C, Z, Y, X = list(feat_input.shape)
        
        out_dict = {}

        x0 = self.encoder_layer0(feat_input)
        x1 = self.encoder_layer1(x0)
        x2 = self.encoder_layer2(x1)
        x3 = self.encoder_layer3(x2)
        x4 = self.encoder_layer4(x3)
        x5 = self.encoder_layer5(x4)
        x6 = self.encoder_layer6(x5)
        x7 = self.encoder_layer7(x6)
        x8 = self.encoder_layer8(x7)
        x9 = self.final_layer(x8)

        if norm:
            feat_norm = utils.basic.l2_normalize(x9, dim=1)

        feat_halfmemX0 = feat_norm[0:1]
        valid_halfmemX0 = torch.ones_like(feat_halfmemX0[:,0:1])

        summ_writer.summ_feat('feat3d/feat_input', feat_input, pca=(C>3))
        summ_writer.summ_feat('feat3d/feat_output', feat_halfmemX0, pca=True)

        occ_e_ = self.occ_conv(feat_halfmemX0)

        out_dict["x0"] = x0
        out_dict["x1"] = x1
        out_dict["x2"] = x2
        out_dict["x3"] = x3
        out_dict["x4"] = x4
        out_dict["x5"] = x5
        out_dict["x6"] = x6
        out_dict["x7"] = x7
        out_dict["x8"] = x8
        out_dict["x9"] = x9
        out_dict["feat_norm"] = feat_norm
        out_dict["occ_e"] = occ_e_
        

        return out_dict

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

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
        
        # self.featnet = FeatNet().cuda()
        # self.feat3dnet.eval()
        # self.set_requires_grad(self.feat3dnet, False)

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

        self.grnn = GRNN().cuda().eval()
        print(self.grnn)

        for p in self.grnn.parameters():
            p.requires_grad = False
        self.grnn.eval()

        model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        self.midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True).eval()

        # midas depth estimation
        # self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS") #, _use_new_zipfile_serialization=True)
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.cuda()
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # self.transform = midas_transforms.default_transform



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
        assert(self.B==1) # need B=1 for this - TODO: allow batching
        self.pix_T_camX = torch.from_numpy(self.pix_T_camX).cuda().unsqueeze(0).repeat(self.B,1,1).float()

        data_loader_transform = transforms.Compose([
                            transforms.ToTensor()])
        dataset = ImageFolderWithPaths(coco_images_path, transform=data_loader_transform) # our custom dataset
        
        # self.dataloader = torch.utils_DataLoader(dataset, batch_size=32, shuffle=False)
        # dataset = datasets.ImageFolder(coco_images_path, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.B, shuffle=False)

        self.Z, self.Y, self.X = Z, Y, X #hyp.Z, hyp.Y, hyp.X

        print("Z:", self.Z, "Y", self.Y, "X", self.X)

        # self.Z = 128
        # self.Y = 128
        # self.X = 128
        # bounds = torch.tensor([-12.0, 12.0, -12.0, 12.0, -12.0, 12.0]).cuda()
        # self.scene_centroid = torch.tensor([0.0, 0.0, 10.0]).unsqueeze(0).repeat(self.B,1).cuda()
        # self.vox_util = utils_vox.Vox_util(self.Z, self.Y, self.X, set_name, scene_centroid=self.scene_centroid, assert_cube=True, bounds=bounds)

        self.writer = SummaryWriter(log_dir + '/' + set_name, max_queue=10, flush_secs=1000)

        # self.avgpool3d = nn.AvgPool3d(2, stride=2)
        self.pool_len = 2
        self.pool3d = nn.AvgPool3d(self.pool_len, stride=self.pool_len)
        self.pool2d = nn.AvgPool2d(self.pool_len, stride=self.pool_len)

        # self.run_extract()
    
    def run_extract(self):

        for layer in layers:
            # if os.path.isfile(f'{output_dir}/{layer}.npy'):
            #     print("LAYER ALREADY EXISTS... SKIPPING")
            #     continue
            idx = 0
            for images, _, file_ids in self.dataloader:

                # if only_process_stim_ids:
                #     if file_ids not in stim_list: 
                #         continue

                print('set name', set_name, 'Layer',layer,'Images processed:', idx)

                self.summ_writer = utils.improc.Summ_writer(
                    writer=self.writer,
                    global_step=idx,
                    log_freq=log_freq,
                    fps=8,
                    # just_gif=True,
                )            

                rgb_camX = images.cuda().float()
                _,_,W_I,H_I = list(rgb_camX.shape)
                rgb_camX_norm = rgb_camX - 0.5
                self.summ_writer.summ_rgb('inputs/rgb', rgb_camX_norm)

                hfov = float(self.fov) * np.pi / 180.
                pix_T_camX = np.array([
                    [(W_I/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
                    [0., (H_I/2.)*1 / np.tan(hfov / 2.), 0., 0.],
                    [0., 0.,  1, 0],
                    [0., 0., 0, 1]])
                pix_T_camX[0,2] = W_I/2. 
                pix_T_camX[1,2] = H_I/2.   
                pix_T_camX = torch.from_numpy(pix_T_camX).cuda().unsqueeze(0).repeat(self.B,1,1).float()
                
                # estimate depth
                rgb_camX = (rgb_camX.permute(0,2,3,1).detach().cpu().numpy() * 255).astype(np.uint8)
                input_batch = []
                for b in range(self.B):
                    input_batch.append(self.transform(rgb_camX[b]).cuda())
                input_batch = torch.cat(input_batch, dim=0)
                with torch.no_grad():
                    depth_cam = self.midas(input_batch).unsqueeze(1)
                    depth_cam = (torch.max(depth_cam) - depth_cam) / 1000.0
                
                self.summ_writer.summ_depth('inputs/depth_map', depth_cam[0].squeeze().detach().cpu().numpy())

                xyz_camXs = utils.geom.depth2pointcloud(depth_cam, pix_T_camX).float()

                xyz_maxs = torch.max(xyz_camXs, dim=1)[0]
                xyz_mins = torch.min(xyz_camXs, dim=1)[0]

                # print(xyz_maxs)
                # print(xyz_mins)

                xyz_max = torch.max(xyz_maxs, dim=1)[0]/2.
                scene_centroid = torch.tensor([0.0, 0.0, xyz_max]).unsqueeze(0).repeat(self.B,1).cuda()
                bounds = torch.tensor([-xyz_max, xyz_max, -xyz_max, xyz_max, -xyz_max, xyz_max]).cuda()
                vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, set_name, scene_centroid, bounds=bounds, assert_cube=False)
                # self.vox_util = utils_vox.Vox_util(self.Z, self.Y, self.X, set_name, self.scene_centroid, bounds=bounds, assert_cube=True)

                occXs = vox_util.voxelize_xyz(xyz_camXs, self.Z, self.Y, self.X)
                unpXs = vox_util.unproject_rgb_to_mem(
                    rgb_camX_norm, self.Z, self.Y, self.X, pix_T_camX)

                self.summ_writer.summ_occ('3D_inputs/occXs', occXs)
                self.summ_writer.summ_unp('3D_inputs/unpXs', unpXs, occXs)
                
                featXs_input = torch.cat([occXs, occXs*unpXs], dim=1)
                # it is useful to keep track of what was visible from each viewpoint
                with torch.no_grad():
                    feats_all = self.grnn(featXs_input, vox_util, pix_T_camX, self.summ_writer)

                # layer_ind = layer_map[layer]
                feat_memX = feats_all[layer]

                num_spatial = len(feat_memX.shape[2:])
                feat_size_flat = np.prod(torch.tensor(feat_memX.shape[1:]).numpy())
                
                if save_subject_pca_features or save_full_features:
                    # 3d pool
                    feat_size_thresh = 800000

                    while feat_size_flat>feat_size_thresh:
                        if num_spatial==3:
                            feat_memX = self.pool3d(feat_memX)
                        elif num_spatial==2:
                            feat_memX = self.pool2d(feat_memX)
                        else:
                            assert(False)
                        feat_size_flat = np.prod(torch.tensor(feat_memX.shape[1:]).numpy())

                    feat_memX = feat_memX.squeeze(0).cpu().numpy().astype(np.float32)
                    print("Feature size:", feat_memX.shape)

                    if idx==0:
                        if num_spatial==3:
                            c,h,w,d = feat_memX.shape
                            feats = np.zeros((73000, c, h, w, d), dtype=np.float32)
                            file_order = np.zeros(73000)
                        elif num_spatial==2:
                            c,h,w = feat_memX.shape
                            feats = np.zeros((73000, c, h, w), dtype=np.float32)
                            file_order = np.zeros(73000)
                        else:
                            assert(False)

                    feats[idx] = feat_memX
                    file_order[idx] = file_ids.detach().cpu().numpy()                

                del feat_memX, featXs_input, occXs, unpXs, xyz_camXs

                idx += 1*self.B

                # plt.close('all')

                # if idx == 10:
                #     break

                if only_process_stim_ids:
                    if idx == 10000:
                        break

            if save_subject_pca_features:
                print("Saving features")

                b = feats.shape[0]
                feats = np.reshape(feats, (b, -1))

                np.save(f"{output_dir}/tmp.npy", feats)
                del feats
                feats = np.load(f"{output_dir}/tmp.npy", mmap_mode='r')

                for subj in subjects:

                    print(f"Saving layer {layer}, subject {subj}")

                    pca_train_file_path =  f"{output_dir}/{layer}_subj{subj}_train_pca.npy"
                    pca_test_file_path =  f"{output_dir}/{layer}_subj{subj}_test_pca.npy"
                    pca_fit_path = f"{output_dir}/{layer}_subj{subj}_pca_fit.p" # path to pca model

                    # if os.path.isfile(pca_train_file_path) and os.path.isfile(pca_test_file_path):
                    #     print(f"{pca_train_file_path} exists. delete old file to regenerate.")
                    #     continue

                    stimulus_list = np.load(
                        "%s/coco_ID_of_repeats_subj%02d.npy" % (stim_dir, subj)
                    )

                    feature_mat = extract_feature_with_image_order(
                        stimulus_list, feats, file_order, 
                    )
                    # feature_mat = feats[:10000]
                    # feature_mat = np.random.rand(10000, feats.shape[1])

                    # split train and test set
                    X_train, X_test = train_test_split(
                        feature_mat, test_size=0.15, random_state=42
                    )

                    del feature_mat

                    _,_ = dimensionality_reduction_feats(
                        X_train, 
                        X_test, 
                        layer, 
                        pca_train_file_path, 
                        pca_test_file_path, 
                        pca_fit_path, 
                        method='pca',
                        # method='incremental_pca', 
                        override_existing_pca=True,
                        n_components=1000,
                        )

                    del X_train, X_test

                os.remove(f"{output_dir}/tmp.npy")
            
            # feats = np.concatenate(feats, axis=0)
            # file_order = np.concatenate(file_order, axis=0)
            dim_red = False
            if dim_red:
                pca = PCA(n_components=1000)

                b,c,h,w,d = feats.shape
                feats = np.reshape(feats, (b, -1))
                feats = pca.fit_transform(feats)

                plt.plot(np.cumsum(pca.explained_variance_ratio_))
                plt.xlabel('number of components')
                plt.ylabel('cumulative explained variance')
                plt.yticks(np.arange(0.0,1.0,0.05))
                plt.savefig(output_dir + '/variance_explained.png')

            if save_full_features:
                np.save(f'{output_dir}/{layer}.npy', feats)
                np.save(f'{output_dir}/file_order.npy', file_order)


if __name__ == '__main__':
    model = CARLA_MOC(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    model.initialize_model()