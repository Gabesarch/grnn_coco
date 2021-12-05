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
from nets.feat3dnet import Feat3dNet


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

layer = 'Res3dBlock8'

# set_name = 'test00'
layer_map = {
    'Res3dBlock1':0, 
    'Pool3dBlock2':1, 
    'Res3dBlock3':2, 
    'Res3dBlock4':3, 
    'Res3dBlock5':4, 
    'Pool3dBlock6':5, 
    'Res3dBlock7':6,
    'Res3dBlock8':7, 
    'Res3dBlock9':8, 
    'Conv3d10':9, 
    }

set_name = 'grnn_contrast_' + layer
print(set_name)

checkpoint_dir='checkpoints/' + set_name
log_dir='logs_grnn_coco'

pool_type = '3d'

# do_viewnet = True
save_feats = True
plot_classes = False
only_process_stim_ids = False
reduce_channels = False
log_freq = 10000 # frequency for logging tensorboard
subj = 1 # subject number - supports: 1, 2, 7
pool_len = 2 # max pool length
num_pool = 1 # number of max pools
pretrain = True
print("SUBJECT:", subj)

if only_process_stim_ids:
    stim_list = np.load(
        "/user_data/yuanw3/project_outputs/NSD/output/coco_ID_of_repeats_subj%02d.npy" % (subj)
    )
    stim_list = list(stim_list)

hyp.view_depth = 32
# hyp.feat_init = '02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_train_ns_grnn00'
hyp.feat3d_init = '01_s2_m128x128x128_1e-4_F3_d32_O_c1_s.1_E3_n2_d16_c1_carla_and_replica_train_cr12'
# if do_viewnet:
#     hyp.view_init = '02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_train_ns_grnn00'

# output_dir = f'/lab_data/tarrlab/gsarch/encoding_model/{set_name}_subj={subj}_pl={pool_len}_nm={num_maxpool}'
# output_dir = '/lab_data/tarrlab/gsarch/encoding_model/grnn_feats_init_32x18x18x18' #/subj%s' % (subj)
output_dir = f'/lab_data/tarrlab/gsarch/encoding_model/{set_name}'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
print("output_dir: ", output_dir)

# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py

# class ImageFolderWithPaths(datasets.ImageFolder):
#     """Custom dataset that includes image file paths. Extends
#     torchvision.datasets.ImageFolder
#     """
#     def __init__(self, coco_images_path, transform):
#         super(ImageFolderWithPaths, self).__init__(coco_images_path, transform=transform)
#         if only_process_stim_ids:
#             image_ids = [int(self.imgs[i][0].split('/')[-1].split('.')[0]) for i in range(len(self.imgs))]
#             idxes = [image_ids.index(cid) for cid in stim_list]
#             self.imgs = [self.imgs[idx] for idx in idxes]

#     # override the __getitem__ method. this is the method that dataloader calls
#     def __getitem__(self, index):
#         # this is what ImageFolder normally returns 
#         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
#         # the image file path
#         path = self.imgs[index][0]
#         path = path.split('/')[-1].split('.')[0]
#         file_id = int(path)

#         # make a new tuple that includes original and the path
#         tuple_with_path = (original_tuple + (file_id,))
#         return tuple_with_path


class GRNN(nn.Module):
    def __init__(self):
        super(GRNN, self).__init__()

        featnet = Feat3dNet(in_dim=4).cuda()

        if pretrain:
            inits = {"featnet": hyp.feat3d_init,}
            print(inits)
            for part, init in list(inits.items()):
                # st()
                if init:
                    if part == 'featnet':
                        model_part = featnet
                    else:
                        assert(False)
                    iter = saverloader.load_part(model_part, part, init)
                    if iter:
                        print("loaded %s at iter %d" % (init, iter))
                    else:
                        print("could not find a checkpoint for %s" % init)

        for p in featnet.parameters():
            p.requires_grad = False
        featnet.eval()

        self.encoder_layer0 = featnet.net.encoder_layer0
        self.encoder_layer1 = featnet.net.encoder_layer1
        self.encoder_layer2 = featnet.net.encoder_layer2
        self.encoder_layer3 = featnet.net.encoder_layer3
        self.encoder_layer4 = featnet.net.encoder_layer4
        self.encoder_layer5 = featnet.net.encoder_layer5
        self.encoder_layer6 = featnet.net.encoder_layer6
        self.encoder_layer7 = featnet.net.encoder_layer7
        self.encoder_layer8 = featnet.net.encoder_layer8
        self.final_layer = featnet.net.final_layer

    
    def forward(self, feat, summ_writer=None, mask=None, rgb_camX_norm=None, pix_T_camX=None, vox_util=None):

        from utils_basic import l2_normalize

        total_loss = torch.tensor(0.0).cuda()
        B, C, D, H, W = list(feat.shape)
        # summ_writer.summ_feat(f'feat/{prefix}feat0_input', feat)

        x0 = self.encoder_layer0(feat)
        x1 = self.encoder_layer1(x0)
        x2 = self.encoder_layer2(x1)
        x3 = self.encoder_layer3(x2)
        x4 = self.encoder_layer4(x3)
        x5 = self.encoder_layer5(x4)
        x6 = self.encoder_layer6(x5)
        x7 = self.encoder_layer7(x6)
        x8 = self.encoder_layer8(x7)
        x9 = self.final_layer(x8)

        return x0, x1, x2, x3, x4, x5, x6, x7, x8, x9

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

        self.featnet = GRNN().cuda().eval()
        print(self.featnet)

        # midas depth estimation
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS") #, _use_new_zipfile_serialization=True)
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.cuda()
        self.midas.eval()

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
        assert(self.B==1) # need B=1 for this - TODO: allow batching
        self.pix_T_camX = torch.from_numpy(self.pix_T_camX).cuda().unsqueeze(0).repeat(self.B,1,1).float()

        data_loader_transform = transforms.Compose([
                            transforms.ToTensor()])
        dataset = ImageFolderWithPaths(coco_images_path, transform=data_loader_transform) # our custom dataset
        
        # self.dataloader = torch.utils_DataLoader(dataset, batch_size=32, shuffle=False)
        # dataset = datasets.ImageFolder(coco_images_path, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.B, shuffle=False)

        self.Z, self.Y, self.X = 128,128,128 #hyp.Z, hyp.Y, hyp.X

        # self.Z = 128
        # self.Y = 128
        # self.X = 128
        bounds = torch.tensor([-12.0, 12.0, -12.0, 12.0, -12.0, 12.0]).cuda()
        self.scene_centroid = torch.tensor([0.0, 0.0, 10.0]).unsqueeze(0).repeat(self.B,1).cuda()
        self.vox_util = utils_vox.Vox_util(self.Z, self.Y, self.X, set_name, scene_centroid=self.scene_centroid, assert_cube=True, bounds=bounds)

        self.writer = SummaryWriter(log_dir + '/' + set_name, max_queue=10, flush_secs=1000)

        # self.avgpool3d = nn.AvgPool3d(2, stride=2)
        self.pool_len = pool_len
        if pool_type=='3d':
            self.pool = nn.AvgPool3d(self.pool_len, stride=self.pool_len)
        elif pool_type=='2d':
            self.pool = nn.AvgPool2d(self.pool_len, stride=self.pool_len)
        else:
            assert(False)


        self.num_pool = num_pool

        # self.run_extract()
    
    def run_extract(self):

        # if reduce_channels:
        #     ch=7
        # else:
        #     ch=32
        
        # spat_dim = 72//(self.pool_len**self.num_pool)
        # if only_process_stim_ids:
        #     feats = np.zeros((10000, ch, spat_dim, spat_dim, spat_dim)).astype(np.float32)
        #     file_order = np.zeros(10000)
        # else:
        #     feats = np.zeros((73000, ch, spat_dim, spat_dim, spat_dim), dtype=np.float16)
        #     file_order = np.zeros(73000)
        cat_names = []
        supcat_names = []
        cat_ids = []
        idx = 0
        for images, _, file_ids in self.dataloader:

            if only_process_stim_ids:
                if file_ids not in stim_list: 
                    continue

            print('Images processed: ', idx)

            self.summ_writer = utils_improc.Summ_writer(
                writer=self.writer,
                global_step=idx,
                set_name=set_name,
                log_freq=log_freq,
                fps=8,
                # just_gif=True,
            )

            rgb_camX = images.cuda().float()
            rgb_camX_norm = rgb_camX - 0.5
            self.summ_writer.summ_rgb('inputs/rgb', rgb_camX_norm)

            if False:
                plt.figure()
                rgb_camXs_np = rgb_camXs[0].permute(1,2,0).detach().cpu().numpy()
                plt.imshow(rgb_camXs_np)
                plt.savefig('images/test.png')


            if plot_classes:
                # get category name
                for b in range(self.B):
                    img_id = [int(file_ids[b].detach().cpu().numpy())]
                    coco_util = self.coco_util_train
                    annotation_ids = coco_util.getAnnIds(img_id)
                    if not annotation_ids:
                        coco_util = self.coco_util_val
                        annotation_ids = coco_util.getAnnIds(img_id)
                    annotations = coco_util.loadAnns(annotation_ids)

                    best_area = 0
                    entity_id = None
                    entity = None
                    for i in range(len(annotations)):
                        if annotations[i]['area'] > best_area:
                            entity_id = annotations[i]["category_id"]
                            entity = coco_util.loadCats(entity_id)[0]["name"]
                            super_cat = coco_util.loadCats(entity_id)[0]["supercategory"]
                            best_area = annotations[i]['area']
                    cat_names.append(entity)
                    cat_ids.append(entity_id)
                    supcat_names.append(super_cat)
            
            # estimate depth
            rgb_camX = (rgb_camX.permute(0,2,3,1).detach().cpu().numpy() * 255).astype(np.uint8)
            input_batch = []
            for b in range(self.B):
                input_batch.append(self.transform(rgb_camX[b]).cuda())
            input_batch = torch.cat(input_batch, dim=0)
            with torch.no_grad():
                depth_cam = self.midas(input_batch).unsqueeze(1)
                depth_cam = (torch.max(depth_cam) - depth_cam) / 100.0

            # depth_cam = self.XTCmodel_depth(rgb_camXs)

            # # get depths in 0-100 range approx.
            # depth_cam = (depth_cam / 0.7) * 100
            
            self.summ_writer.summ_depth('inputs/depth_map', depth_cam[0].squeeze().detach().cpu().numpy())

            xyz_camXs = utils_geom.depth2pointcloud(depth_cam, self.pix_T_camX).float()

            xyz_maxs = torch.max(xyz_camXs, dim=1)[0]
            xyz_mins = torch.min(xyz_camXs, dim=1)[0]

            # shift_am = torch.tensor([(xyz_maxs[0][0] - torch.abs(xyz_mins[0][0]))/2., 0., 0.]).cuda().unsqueeze(0).unsqueeze(0)
            # xyz_camXs = xyz_camXs - shift_am
            # xyz_max = torch.max(xyz_camXs)
            xyz_max = torch.max(xyz_maxs, dim=1)[0]/2.
            self.scene_centroid = torch.tensor([0.0, 0.0, xyz_max]).unsqueeze(0).repeat(self.B,1).cuda()
            bounds = torch.tensor([-xyz_max, xyz_max, -xyz_max, xyz_max, -xyz_max, xyz_max]).cuda()
            self.vox_util = utils_vox.Vox_util(self.Z, self.Y, self.X, set_name, self.scene_centroid, bounds=bounds, assert_cube=True)

            # occ_memX = self.vox_util.voxelize_xyz(xyz_camXs, self.Z, self.Y, self.X)
            # self.summ_writer.summ_occ('inputs/occ_memX', occ_memX)
            # unp_memX = self.vox_util.unproject_rgb_to_mem(
            #     rgb_camX_norm, self.Z, self.Y, self.X, self.pix_T_camX)
            # feat_memX_input = torch.cat([occ_memX, occ_memX*unp_memX], dim=1)
            # feat3d_loss, feat_memX, _ = self.feat3dnet(
            #     feat_memX_input,
            #     self.summ_writer,
            # )
            occXs = self.vox_util.voxelize_xyz(xyz_camXs, self.Z, self.Y, self.X)
            unpXs = self.vox_util.unproject_rgb_to_mem(
                rgb_camX_norm, self.Z, self.Y, self.X, self.pix_T_camX)


            self.summ_writer.summ_occ('3D_inputs/occXs', occXs)
            self.summ_writer.summ_unp('3D_inputs/unpXs', unpXs, occXs)
            
            featXs_input = torch.cat([occXs, occXs*unpXs], dim=1)
            # it is useful to keep track of what was visible from each viewpoint
            # conv1_out, conv2_out, conv3_out, conv3d_transpose1_out, conv3d_transpose2_out, final_feature_out, feat_memX, view_pool_out, view_conv3d_out, view_conv2d1_out, view_conv2d2_out, view_final_conv_out, view_emb_layer_out, view_rgb_layer_out = self.featnet(featXs_input, self.summ_writer, mask=occXs, rgb_camX_norm=rgb_camX_norm, pix_T_camX=self.pix_T_camX, vox_util=self.vox_util)
            feats_all = self.featnet(featXs_input, self.summ_writer, mask=occXs, rgb_camX_norm=rgb_camX_norm, pix_T_camX=self.pix_T_camX, vox_util=self.vox_util)

            layer_ind = layer_map[layer]
            feat_memX = feats_all[layer_ind]

            if save_feats:
                # 3d pool
                if self.num_pool>0:
                    for nm in range(self.num_pool):
                        feat_memX = self.pool(feat_memX)

                feat_memX = feat_memX.squeeze(0).detach().cpu().numpy().astype(np.float16)
                
                if reduce_channels:
                    c,h,w,d = feat_memX.shape
                    feat_memX = np.transpose(feat_memX, (1,2,3,0))
                    feat_memX = np.reshape(feat_memX, (h*w*d, c))
                    pca = PCA(n_components=ch)
                    feat_memX = pca.fit_transform(feat_memX)
                    # print(feat_memX.shape)
                    feat_memX = np.reshape(feat_memX, (h,w,d,ch))
                    feat_memX = np.transpose(feat_memX, (3,0,1,2))

                    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
                    # plt.xlabel('number of components')
                    # plt.ylabel('cumulative explained variance')
                    # plt.yticks(np.arange(0.0,1.0,0.05))
                    # plt.savefig('images/variance_explained.png')
                    # st()

                # feats.append(feat_memX.detach().cpu().numpy())
                # file_order.append(file_ids.detach().cpu().numpy())
                # st()
                if idx==0:
                    if pool_type == '3d':
                        c,h,w,d = feat_memX.shape
                        feats = np.zeros((73000, c, h, w, d), dtype=np.float16)
                        file_order = np.zeros(73000)
                    elif pool_type == '2d':
                        c,h,w = feat_memX.shape
                        feats = np.zeros((73000, c, h, w), dtype=np.float16)
                        file_order = np.zeros(73000)

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

        if save_feats:
            np.save(f'{output_dir}/feats.npy', feats)
            np.save(f'{output_dir}/file_order.npy', file_order)
            if plot_classes:
                np.save(f'{output_dir}/supcat.npy', np.array(supcat_names))
                np.save(f'{output_dir}/cat.npy', np.array(cat_names))

        if plot_classes:
            feats = np.reshape(feats, (feats.shape[0], -1))

            tsne = TSNE(n_components=2).fit_transform(feats)
            # pred_catnames_feats = [self.maskrcnn_to_ithor[i] for i in self.feature_obj_ids]

            self.plot_by_classes(supcat_names, tsne, self.summ_writer)

            # tsne plot colored by predicted labels
            tsne_pred_figure = self.get_colored_tsne_image(supcat_names, tsne)
            self.summ_writer.summ_figure(f'tsne/tsne_grnn_reduced_supcat', tsne_pred_figure)

            tsne_pred_figure = self.get_colored_tsne_image(cat_names, tsne)
            self.summ_writer.summ_figure(f'tsne/tsne_grnn_reduced_cat', tsne_pred_figure)

    # def plot_tsne_only(self):

    #     self.summ_writer = utils_improc.Summ_writer(
    #             writer=self.writer,
    #             global_step=1,
    #             log_freq=1,
    #             fps=8,
    #             just_gif=True,
    #         )

    #     feats = np.load(f'{output_dir}/replica_carla_feats.npy')
    #     # file_order = np.load(f'{output_dir}/replica_carla_file_order.npy')
    #     supcat_names = np.load(f'{output_dir}/replica_carla_supcat.npy')
    #     cat_names = np.load(f'{output_dir}/replica_carla_cat.npy')

    #     tsne = TSNE(n_components=2).fit_transform(feats)
    #     # pred_catnames_feats = [self.maskrcnn_to_ithor[i] for i in self.feature_obj_ids]
            

    #     # # tsne plot colored by predicted labels
    #     # tsne_pred_figure = self.get_colored_tsne_image(supcat_names, tsne)
    #     # self.summ_writer.summ_figure(f'tsne/tsne_grnn_reduced_supcat', tsne_pred_figure)

    #     # tsne_pred_figure = self.get_colored_tsne_image(cat_names, tsne)
    #     # self.summ_writer.summ_figure(f'tsne/tsne_grnn_reduced_cat', tsne_pred_figure)

    #     self.plot_by_classes(supcat_names, tsne, self.summ_writer)
        
        

    #     # if as_image: 
    #     #     image = self.plot_to_image(figure)
    #     #     return image
    #     # else:
    #     #     return figure


    # def plot_depth(self):

    #     self.summ_writer = utils_improc.Summ_writer(
    #             writer=self.writer,
    #             global_step=1,
    #             log_freq=1,
    #             fps=8,
    #             just_gif=True,
    #         )

    #     feats = np.load(f'{output_dir}/replica_carla_feats.npy')#[:10]
    #     # file_order = np.load(f'{output_dir}/replica_carla_file_order.npy')
    #     supcat_names = np.load(f'{output_dir}/replica_carla_supcat.npy')#[:10]
    #     cat_names = np.load(f'{output_dir}/replica_carla_cat.npy')#[:10]
        


    #     median = []
    #     mean = []
    #     idx = 0
    #     for images, _, file_ids in self.dataloader:

    #         print('Images processed: ', idx)

    #         self.summ_writer = utils_improc.Summ_writer(
    #             writer=self.writer,
    #             global_step=idx,
    #             log_freq=50,
    #             fps=8,
    #             just_gif=True,
    #         )

    #         rgb_camX = images.cuda().float()
            
    #         # estimate depth
    #         rgb_camX = (rgb_camX.permute(0,2,3,1).detach().cpu().numpy() * 255).astype(np.uint8)
    #         input_batch = []
    #         for b in range(self.B):
    #             input_batch.append(self.transform(rgb_camX[b]).cuda())
    #         input_batch = torch.cat(input_batch, dim=0)
    #         with torch.no_grad():
    #             depth_cam = self.midas(input_batch).unsqueeze(1)
    #             depth_cam = (torch.max(depth_cam) - depth_cam) / 100.0


    #         # for b in range(self.B):
    #         depth_cam = depth_cam - torch.min(depth_cam.reshape(self.B, -1), dim=1).values.reshape(-1, 1, 1, 1)
    #         depth_cam = depth_cam / torch.max(depth_cam.reshape(self.B, -1), dim=1).values.reshape(-1, 1, 1, 1)

    #         median.append(torch.median(depth_cam.reshape(self.B, -1), dim=1).values)
    #         mean.append(torch.mean(depth_cam.reshape(self.B, -1), dim=1))

    #         idx += 1*self.B

    #         # if idx == 100:
    #         #     break

    #     median = torch.cat(median, dim=0)
    #     mean = torch.cat(mean, dim=0)
    #     data_to_plot = torch.stack([median, mean]).t().detach().cpu().numpy()

    #     tx = data_to_plot[:, 0]
    #     ty = data_to_plot[:, 1]
    #     # tx = self.scale_to_01_range(tx)
    #     # ty = self.scale_to_01_range(ty)

    #     unique_classes = []
    #     unique_classes.extend(supcat_names)
    #     unique_classes = list(set(unique_classes))

    #     markers = ['^', 'X', 'o', 's', 'p']
    #     num_markers = len(markers)

    #     # predicted clusters
    #     evenly_spaced_interval = np.linspace(0, 1, len(unique_classes))
    #     colors = [cm.gist_rainbow(x) for x in evenly_spaced_interval]
    #     for idx in range(len(unique_classes)):
    #         figure = plt.figure(figsize = (20,20))
    #         ax = figure.add_subplot(111)

    #         label = unique_classes[idx]
    #         indices = [i for i, l in enumerate(supcat_names) if l == label]

    #         current_tx = np.take(tx, indices)
    #         # current_ty = np.take(ty, indices)

    #         marker = markers[idx%num_markers]
    #         color = colors[idx]
    #         ax.hist(current_tx, bins=np.arange(0,1,0.05))
    #         plt.xlabel('median')
    #         # ax.ylabel('mean')
    #         # ax.scatter(current_tx, current_ty, c=color, marker=marker, label=label)

    #         # ax.legend(loc='best')

    #         # if x_label is not None:
    #         #     plt.xlabel(x_label)
    #         # if y_label is not None:
    #         #     plt.ylabel(y_label)

    #         self.summ_writer.summ_figure(f'depth_supcat/depth_median_{label}', figure)

    #         plt.close()

    #     # # tsne plot colored by predicted labels
    #     # tsne_pred_figure = self.get_colored_tsne_image(supcat_names, data_to_plot)
    #     # plt.xlabel('median')
    #     # plt.ylabel('mean')
    #     # self.summ_writer.summ_figure(f'depth/depth_supcat', tsne_pred_figure)

    #     # tsne_pred_figure = self.get_colored_tsne_image(cat_names, data_to_plot)
    #     # plt.xlabel('median')
    #     # plt.ylabel('mean')
    #     # self.summ_writer.summ_figure(f'depth/depth_cat', tsne_pred_figure)

    #     # self.plot_by_classes(supcat_names, data_to_plot, self.summ_writer, x_label='median', y_label='mean')


    # def plot_by_classes(self, catnames, data_to_plot, summ_writer, x_label=None, y_label=None):

    #     tx = data_to_plot[:, 0]
    #     ty = data_to_plot[:, 1]
    #     tx = self.scale_to_01_range(tx)
    #     ty = self.scale_to_01_range(ty)

    #     unique_classes = []
    #     unique_classes.extend(catnames)
    #     unique_classes = list(set(unique_classes))

    #     markers = ['^', 'X', 'o', 's', 'p']
    #     num_markers = len(markers)

    #     # predicted clusters
    #     evenly_spaced_interval = np.linspace(0, 1, len(unique_classes))
    #     colors = [cm.gist_rainbow(x) for x in evenly_spaced_interval]
    #     for idx in range(len(unique_classes)):
    #         figure = plt.figure(figsize = (20,20))
    #         ax = figure.add_subplot(111)

    #         label = unique_classes[idx]
    #         indices = [i for i, l in enumerate(catnames) if l == label]

    #         current_tx = np.take(tx, indices)
    #         current_ty = np.take(ty, indices)

    #         marker = markers[idx%num_markers]
    #         color = colors[idx]
    #         ax.scatter(current_tx, current_ty, c=color, marker=marker, label=label)

    #         ax.legend(loc='best')

    #         if x_label is not None:
    #             plt.xlabel(x_label)
    #         if y_label is not None:
    #             plt.ylabel(y_label)

    #         summ_writer.summ_figure(f'depth_classes/depth_supcat_{label}', figure)

    #         plt.close()


    # def get_colored_tsne_image(self, catnames, tsne, as_image=False):
    #     tx = tsne[:, 0]
    #     ty = tsne[:, 1]
    #     tx = self.scale_to_01_range(tx)
    #     ty = self.scale_to_01_range(ty)

    #     figure = plt.figure(figsize = (20,20))
    #     ax = figure.add_subplot(111)

    #     unique_classes = []
    #     unique_classes.extend(catnames)
    #     unique_classes = list(set(unique_classes))

    #     markers = ['^', 'X', 'o', 's', 'p']
    #     num_markers = len(markers)

    #     # predicted clusters
    #     evenly_spaced_interval = np.linspace(0, 1, len(unique_classes))
    #     colors = [cm.gist_rainbow(x) for x in evenly_spaced_interval]
    #     for idx in range(len(unique_classes)):
    #         label = unique_classes[idx]
    #         indices = [i for i, l in enumerate(catnames) if l == label]

    #         current_tx = np.take(tx, indices)
    #         current_ty = np.take(ty, indices)

    #         marker = markers[idx%num_markers]
    #         color = colors[idx]
    #         ax.scatter(current_tx, current_ty, c=color, marker=marker, label=label)
        
    #     ax.legend(loc='best')

    #     if as_image: 
    #         image = self.plot_to_image(figure)
    #         return image
    #     else:
    #         return figure

    # def scale_to_01_range(self,x):
    #     value_range = (np.max(x) - np.min(x))
    #     starts_from_zero = x - np.min(x)
    #     return starts_from_zero / value_range


if __name__ == '__main__':
    model = CARLA_MOC(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    model.initialize_model()
