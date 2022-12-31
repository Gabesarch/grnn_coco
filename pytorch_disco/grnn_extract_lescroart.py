import os
# script = """
MODE="LESCROART_MOC"
# export MODE
# """
# os.system("bash -c '%s'" % script)
os.environ["MODE"] = MODE
os.environ["exp_name"] = 'trainer_lescroart_feat3d_enc3d_occ_emb3d_vox'
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
# from nets.feat3dnet import Feat3dNet

from nets.feat3dnet import Feat3dNet
# from nets.emb3dnet import Emb3dNet
from nets.occnet import OccNet
from nets.viewnet import ViewNet

from attrdict import AttrDict


import cv2

from tensorboardX import SummaryWriter
import torch.nn.functional as F

from backend.inputs_lescroart import MuJoCoOfflineData
from tqdm import tqdm

# import utils_samp
# import utils_geom
# import utils_improc
# import utils_basic
# import utils_eval
# # import utils_py
# import utils_misc
# # import utils_track
# import utils_vox_other as utils_vox

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

import utils.lescroart_data_utils as utils_lescroart


import sys
# sys.path.append("XTConsistency")
# from modules.unet import UNet, UNetReshade


# np.set_printoptions(precision=2)
# np.random.seed(0)

import ipdb
st = ipdb.set_trace

# do_feat3d = True

# XTC_init = '/home/gsarch/repo/pytorch_disco/saved_checkpoints/XTC_checkpoints/rgb2depth_consistency_wimagenet.pth'
coco_images_path = '/lab_data/tarrlab/common/datasets/NSD_images'

tag_sizes = {
    'trn1': 9000,
    'trn2': 9000,
    'trn3': 9000,
    'trn4': 9000,
    'test': 2700,
    }

# layer = 'Res3dBlock8'

# set_name = 'test00'
layers = [
    'x9',
    'feat_norm',
    'x0',
    'x1',
    'x2',
    'x3',
    'x4',
    'x5',
    'x6',
    'x7',
    'x8',
    'occ_e',
]
set_name = 'markdata_view_01'
set_name = 'markdata_emb3d_02'
# set_name = 'markdata_enc3d_emb3d_07_repcarl_camX' # encode as camX
# set_name = 'markdata_enc3d_emb3d_07_repcarl_camX0' # encode as canX0
set_name = 'markdata_enc3d_emb3d_07_repcarl_camX_noavg' # encode as camX
set_name = 'markdata_enc3d_emb3d_07_repcarl_camX0_noavg' # encode as canX0
set_name = 'carl_rep_lesc05_emb3d_camX0_noavg' # encode as canX0
print(set_name)

checkpoint_dir='checkpoints/' + set_name
log_dir='logs_carla_moc'

# do_viewnet = True
use_camX0 = True
flip_layer_order = False
save_feats = True
plot_classes = False
only_process_stim_ids = False
reduce_channels = False
log_freq = 10000 # frequency for logging tensorboard
# subj = 1 # subject number - supports: 1, 2, 7
pretrain = True
block_average = False
block_average_len = 30 # 30 stimulus frames per TR
zscore_after_block_averaging = False # z score features after block averaging
# print("SUBJECT:", subj)
print("BLOCK AVERAGING?", block_average)

if flip_layer_order:
    layers = list(np.flip(np.array(layers)))

if only_process_stim_ids:
    stim_list = np.load(
        "/user_data/yuanw3/project_outputs/NSD/output/coco_ID_of_repeats_subj%02d.npy" % (subj)
    )
    stim_list = list(stim_list)
# mode = 'view'
if set_name=='markdata_enc3d_emb3d_07_repcarl':
    hyp.feat3d_init = '01_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_ns_enc3d_emb3d_07_fitvox'
    # hyp.view_init = '01_m144x144x144_p128x384_1e-4_O_c1_s.1_V_d32_c1_carla_and_replica_train_ns_enc3d_view07_fitvox'
    hyp.occ_init = '01_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_ns_enc3d_emb3d_07_fitvox'
elif "emb3d" in set_name:
    # emb3d
    # hyp.feat3d_init = "01_m144x144x144_2e-5_O_c1_s1_mark_data_train_mark_data_train_ns_enc3d_emb3d_02_fitvox"
    # hyp.view_init = ""
    # hyp.occ_init = "01_m144x144x144_2e-5_O_c1_s1_mark_data_train_mark_data_train_ns_enc3d_emb3d_02_fitvox"
    hyp.feat3d_init = "03_m144x144x144_2e-5_O_c1_s.1_ns_carl_rep_lesc05"
    hyp.view_init = ""
    hyp.occ_init = "03_m144x144x144_2e-5_O_c1_s.1_ns_carl_rep_lesc05"
    
elif "view" in set_name:
    # view
    hyp.feat3d_init = "01_m144x144x144_p128x384_2e-5_O_c1_s1_V_d32_c1_mark_data_train_mark_data_train_ns_enc3d_view_01_fitvox"
    hyp.view_init = "01_m144x144x144_p128x384_2e-5_O_c1_s1_V_d32_c1_mark_data_train_mark_data_train_ns_enc3d_view_01_fitvox"
    hyp.occ_init = "01_m144x144x144_p128x384_2e-5_O_c1_s1_V_d32_c1_mark_data_train_mark_data_train_ns_enc3d_view_01_fitvox"
else:
    assert(False)

# output_dir = f'/lab_data/tarrlab/gsarch/encoding_model/{set_name}_subj={subj}_pl={pool_len}_nm={num_maxpool}'
# output_dir = '/lab_data/tarrlab/gsarch/encoding_model/grnn_feats_init_32x18x18x18' #/subj%s' % (subj)
# output_dir = f'/lab_data/tarrlab/gsarch/encoding_model/{set_name}'
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

        # # self.W, self.H = 256, 256
        # self.fov = 90
        # self.hfov = float(self.fov) * np.pi / 180.
        # self.pix_T_camX = np.array([
        #     [(self.W/2.)*1 / np.tan(self.hfov / 2.), 0., 0., 0.],
        #     [0., (self.H/2.)*1 / np.tan(self.hfov / 2.), 0., 0.],
        #     [0., 0.,  1, 0],
        #     [0., 0., 0, 1]])
        # self.pix_T_camX[0,2] = self.W/2. 
        # self.pix_T_camX[1,2] = self.H/2.         

        # self.B = 1 
        # # assert(self.B==1) # need B=1 for this - TODO: allow batching
        # self.pix_T_camX = torch.from_numpy(self.pix_T_camX).cuda().unsqueeze(0).repeat(self.B,1,1).float()

    
    def forward(self, feat_input, summ_writer=None, norm=True):

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

        # # midas depth estimation
        # self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS") #, _use_new_zipfile_serialization=True)
        # # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.midas.cuda()
        # self.midas.eval()

        # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # self.transform = midas_transforms.default_transform



        # # self.W = 256
        # # self.H = 256
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
        

        self.B = 1 
        assert(self.B==1) # need B=1 for this - TODO: allow batching
        # self.pix_T_camX = torch.from_numpy(self.pix_T_camX).cuda().unsqueeze(0).repeat(self.B,1,1).float()

        # data_loader_transform = transforms.Compose([
        #                     transforms.ToTensor()])
        # dataset = ImageFolderWithPaths(coco_images_path, transform=data_loader_transform) # our custom dataset
        
        # # self.dataloader = torch.utils_DataLoader(dataset, batch_size=32, shuffle=False)
        # # dataset = datasets.ImageFolder(coco_images_path, transform=transform)
        # self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.B, shuffle=False)

        # hyp.S = 1
        # hyp.V = 1
        # self.dataloader = torch.utils.data.DataLoader(
        #                     dataset=MuJoCoOfflineData(hyp, dataset_path='/lab_data/tarrlab/common/datasets/markdata/mark_testdata_test.txt', plot=False, num_workers=1, preprocess_on_batch=True, dataset_name="markdata_test"),
        #                     shuffle=False,
        #                     batch_size=self.B,
        #                     num_workers=1,
        #                     pin_memory=True,
        #                     drop_last=True,
        #                 )
        self.tags = ['trn1', 'trn2', 'trn3', 'trn4', 'test']

        self.data_root = '/lab_data/tarrlab/common/datasets/markdata/'

        filename_dict_ordered = {}
        for tag in self.tags:

            tag_file = os.path.join(self.data_root, f'mark_testdata_{tag}.txt')
            with open(tag_file, 'r') as f:
                all_files = f.readlines()

            for a in range(len(all_files)):
                all_files[a] = all_files[a][:-1] # remove /n

            val_order = []
            for file_ in all_files:
                val_order.append(int(file_[-11:-7]))
            
            all_files = np.array(all_files)
            filename_dict_ordered[tag] = list(all_files[np.argsort(np.array(val_order))])


        # st()




        # # process files in correct order
        # trn_file = os.path.join(self.data_root, 'mark_data_train.txt')
        # with open(trn_file, 'r') as f:
        #     all_files = f.readlines()
        

        # for a in range(len(all_files)):
        #     all_files[a] = all_files[a][:-1] # remove /n

        # filename_dict = {}
        # filename_dict_ordered = {}
        # for tag in trn_tags:
        #     filename_dict[tag] = {}
        #     filename_dict[tag]["filenames"] = []
        #     filename_dict[tag]["order"] = []
        #     filename_dict_ordered[tag] = []

        # for file_ in all_files:
        #     for tag in trn_tags:
        #         if tag in file_:
        #             filename_dict[tag]["filenames"].append(file_) # remove /n
        #             filename_dict[tag]["order"].append(int(file_[-11:-7]))
        #             break
        
        # for tag in trn_tags:
        #     fnames = np.array(filename_dict[tag]["filenames"])
        #     filename_dict_ordered[tag] = list(fnames[np.argsort(np.array(filename_dict[tag]["order"]))])
            
        # # data_trn = np.load(trn_file, allow_pickle=True).item()
        # val_file = os.path.join(self.data_root, 'mark_testdata_test.txt')
        # with open(val_file, 'r') as f:
        #     all_files = f.readlines()

        # for a in range(len(all_files)):
        #     all_files[a] = all_files[a][:-1] # remove /n

        # val_order = []
        # for file_ in all_files:
        #     val_order.append(int(file_[-11:-7]))
        
        # all_files = np.array(all_files)
        # filename_dict_ordered['val'] = list(all_files[np.argsort(np.array(val_order))])


        # data_val = np.load(trn_file, allow_pickle=True).item()

        # val_sequence = utils.load_data(os.path.join('/lab_data/tarrlab/common/datasets/Lescroart2018_fmri', 'validation_sequence.hdf'), 'sequence')
        # st()

        # self.tags = ['trn1', 'trn2', 'trn3', 'trn4', 'val']
        self.filename_dict_ordered = filename_dict_ordered

        self.dataset_name = "markdata"

        if self.dataset_name == "markdata":
            mujoco_T_adam = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
            origin_T_adam = np.zeros((4,4), dtype=np.float32)
            origin_T_adam[:3, :3] = mujoco_T_adam
            #origin_T_adam[:3, 3] = origin_T_camR_xpos
            origin_T_adam[3,3] = 1
            self.origin_T_adam = origin_T_adam
            self.adam_T_origin = np.linalg.inv(self.origin_T_adam)

        # brain_data_dir = '/lab_data/tarrlab/common/datasets/Lescroart2018_fmri/'
        # r1 = utils_lescroart.load_data(os.path.join(brain_data_dir, 'subject01_fmri_data_trn.hdf'), 'run1')
        # r2 = utils_lescroart.load_data(os.path.join(brain_data_dir, 'subject01_fmri_data_trn.hdf'), 'run2')
        # r3 = utils_lescroart.load_data(os.path.join(brain_data_dir, 'subject01_fmri_data_trn.hdf'), 'run3')
        # r4 = utils_lescroart.load_data(os.path.join(brain_data_dir, 'subject01_fmri_data_trn.hdf'), 'run4')
        # st()

        ######## VISUALIZE VIDEO ###########
        # image_folder = '/lab_data/tarrlab/common/datasets/Lescroart2018_fmri/stimuli_val'
        # video_name = 'images/video.avi'

        # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        # frame = cv2.imread(os.path.join(image_folder, images[0]))
        # height, width, layers = frame.shape

        # video = cv2.VideoWriter(video_name, 0, 15, (width,height))

        # for image in images:
        #     video.write(cv2.imread(os.path.join(image_folder, image)))

        # cv2.destroyAllWindows()
        # video.release()
        # st()
        #####################################

        # data = np.load(os.path.join(self.data_root, filename_dict_ordered['val'][0]), allow_pickle=True).item()

        # we wont use a dataloader here

        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X

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

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        # we do one layer at a time because saving all layer features at once is too memory costly
        for layer in tqdm(layers):
            for tag in tqdm(self.tags, leave=False):
                layer_save_name = f'{layer}_{tag}'
                if os.path.isfile(f'{output_dir}/{layer_save_name}.npy'):
                    print("LAYER ALREADY EXISTS... SKIPPING")
                    continue
                filenames = self.filename_dict_ordered[tag]
                idx = 0
                # total_size = 0
                print('Layer',layer)
                for filename in tqdm(filenames, leave=False):

                    # print(set_name)

                    data = self.get_processed_data(filename)
                    # size_ = data['rgb_camXs'].shape[1]
                    # total_size += size_
                    # continue
                    self.prepare_common_tensors(data)

                    # if only_process_stim_ids:
                    #     if file_ids not in stim_list: 
                    #         continue

                    # print('Layer',layer,'Images processed:', idx)

                    self.summ_writer = utils.improc.Summ_writer(
                        writer=self.writer,
                        global_step=idx,
                        log_freq=log_freq,
                        fps=8,
                        # just_gif=True,
                    )         

                    S = self.xyz_camXs.shape[1]

                    for s in range(S):
                        
                        if use_camX0:
                            # print("USING CAMX0")
                            occX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs[:,s:s+1]), self.Z, self.Y, self.X))
                            # unp_memX0 = self.vox_util.unproject_rgb_to_mem(
                            #     __p(self.rgb_camXs), self.Z, self.Y, self.X, self.pix_T_cams[:,0])
                            unpXs = __u(self.vox_util.unproject_rgb_to_mem(
                                __p(self.rgb_camXs[:,s:s+1]), self.Z, self.Y, self.X, __p(self.pix_T_cams[:,s:s+1])))

                            unpX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs[:,s:s+1], unpXs)
                            featXs_input = torch.cat([occX0s, occX0s*unpX0s], dim=2)
                        else:
                            # print("USING CAMX")
                            occXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs[:,s:s+1]), self.Z, self.Y, self.X))
                            # unp_memX0 = self.vox_util.unproject_rgb_to_mem(
                            #     __p(self.rgb_camXs), self.Z, self.Y, self.X, self.pix_T_cams[:,0])
                            unpXs = __u(self.vox_util.unproject_rgb_to_mem(
                                __p(self.rgb_camXs[:,s:s+1]), self.Z, self.Y, self.X, __p(self.pix_T_cams[:,s:s+1])))
                            # feat_memX0_input = torch.cat([occ_memX0, occ_memX0*unp_memX0], dim=1)
                            featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)
                        
                        featXs_input_ = __p(featXs_input)
                            
                        # it is useful to keep track of what was visible from each viewpoint
                        with torch.no_grad():
                            feats_all = self.grnn(featXs_input_, self.summ_writer)

                        # layer_ind = layer_map[layer]
                        feat_memX = feats_all[layer]

                        num_spatial = len(feat_memX.shape[2:])
                        feat_size_flat = np.prod(torch.tensor(feat_memX.shape[1:]).numpy())
                        
                        if save_feats:
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
                                    feats = np.zeros((feature_size, c, h, w, d), dtype=np.float32)
                                    # file_order = np.zeros(73000)
                                elif num_spatial==2:
                                    c,h,w = feat_memX.shape
                                    feats = np.zeros((feature_size, c, h, w), dtype=np.float32)
                                    # file_order = np.zeros(73000)
                                else:
                                    assert(False)

                            feats[idx] = feat_memX
                            # file_order[idx] = file_ids.detach().cpu().numpy()     

                            idx += 1*self.B
                
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

                # print(tag, total_size)

                if save_feats:
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
                    print("Saving...")
                    np.save(f'{output_dir}/{layer_save_name}.npy', feats)
                    # np.save(f'{output_dir}/file_order.npy', file_order)
                    del feats
    
    def prepare_common_tensors(self, feed):
        results = dict()
        
        # if prep_summ:
        #     self.summ_writer = utils.improc.Summ_writer(
        #         writer=feed['writer'],
        #         global_step=feed['global_step'],
        #         log_freq=feed['set_log_freq'],
        #         fps=8,
        #         just_gif=feed['just_gif'],
        #     )
        # else:
        #     self.summ_writer = None

        # self.include_vis = True

        # self.B = feed["set_batch_size"]
        # self.S = feed["set_seqlen"]

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        
        # self.rgb_camRs = feed["rgb_camRs"]
        self.rgb_camXs = feed["rgb_camXs"].float()
        self.pix_T_cams = feed["pix_T_cams"].float()

        if 'origin_T_camRs' in feed:
            self.origin_T_camRs = feed["origin_T_camRs"]
        elif 'camR_T_origin'in feed:
            self.origin_T_camRs = __u(utils.geom.safe_inverse(__p(feed["camR_T_origin"]))).float()
        elif 'camRs_T_origin'in feed:
            self.origin_T_camRs = __u(utils.geom.safe_inverse(__p(feed["camRs_T_origin"]))).float()
        else:
            assert(False) # need camR info
        self.origin_T_camXs = feed["origin_T_camXs"].float()

        self.camX0s_T_camXs = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0 = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(utils.geom.safe_inverse(__p(self.origin_T_camRs)), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(utils.geom.safe_inverse(__p(self.camRs_T_camXs)))
        self.camXs_T_camX0s = __u(utils.geom.safe_inverse(__p(self.camX0s_T_camXs)))
        self.camX0_T_camR0 = utils.basic.matmul2(self.camX0s_T_camXs[:,0], self.camXs_T_camRs[:,0])
        self.camR0s_T_camXs = utils.basic.matmul2(self.camR0s_T_camRs, self.camRs_T_camXs)
        
        # self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        # override H and W
        self.H = self.rgb_camXs.shape[-2]
        self.W = self.rgb_camXs.shape[-1]
        self.PH, self.PW = hyp.PH, hyp.PW
        self.K = hyp.K
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z1, self.Y1, self.X1 = int(self.Z/1), int(self.Y/1), int(self.X/1)
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.xyz_camXs = feed["xyz_camXs"].float()
        self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camR0s = __u(utils.geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))

        # if feed['set_name']=='test':
        #     self.box_camRs = feed["box_traj_camR"]
        #     # box_camRs is B x S x 9
        #     self.score_s = feed["score_traj"]
        #     self.tid_s = torch.ones_like(self.score_s).long()
        #     self.lrt_camRs = utils.geom.convert_boxlist_to_lrtlist(self.box_camRs)
        #     self.lrt_camXs = utils.geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
        #     self.lrt_camX0s = utils.geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)
        #     self.lrt_camR0s = utils.geom.apply_4x4s_to_lrts(self.camR0s_T_camRs, self.lrt_camRs)

        # if hyp.do_midas_depth_estimation:

        #     # estimate depth
        #     rgb_camXs_midas = ((self.rgb_camXs.view(self.B*self.S,3,self.W,self.H).permute(0,2,3,1).detach().cpu().numpy()+0.5) * 255).astype(np.uint8)
        #     input_batch = []
        #     for b in range(self.B):
        #         input_batch.append(self.transform(rgb_camX[b]).cuda())
        #     input_batch = torch.cat(input_batch, dim=0)
        #     with torch.no_grad():
        #         depth_cam = self.midas(input_batch).unsqueeze(1)
        #         depth_cam = (torch.max(depth_cam) - depth_cam) / 1000.0
        # depth_at_center, center_to_boundary = (feed["depth_at_center"], feed["center_to_boundary"])
        # scene_centroid_x_noise = np.random.normal(0, 0.2)
        # scene_centroid_y_noise = np.random.normal(0, 0.2)
        # scene_centroid_z_noise = np.random.normal(0, 0.2)
        # self.scene_centroid = torch.tensor([0.0+scene_centroid_x_noise, 0.0+scene_centroid_y_noise, depth_at_center+scene_centroid_z_noise]).unsqueeze(0).repeat(self.B,1).cuda()
        # bounds = torch.tensor([-xyz_max, xyz_max, -xyz_max, xyz_max, -xyz_max, xyz_max]).cuda()
        # -center_to_boundary, center_to_boundary, -center_to_boundary, center_to_boundary, depth_at_center - center_to_boundary, depth_at_center + center_to_boundary, 0.0, -0.4

        self.vox_utils = []
        # if "depth_at_center" in feed.keys():
        #     st()
        #     offset_x, offset_y, offset_z = feed["depth_at_center"]#.cpu().numpy()
        # else:
        #     offset_x, offset_y, offset_z = 0., 0., 0. 
        # for b in range(self.B):
            
        xyz_maxs = torch.max(self.xyz_camXs[:,0], dim=1)[0]
        xyz_mins = torch.min(self.xyz_camXs[:,0], dim=1)[0]

        # print(xyz_maxs)
        # print(xyz_mins)

        # shift_am = torch.tensor([(xyz_maxs[0][0] - torch.abs(xyz_mins[0][0]))/2., 0., 0.]).cuda().unsqueeze(0).unsqueeze(0)
        # xyz_camXs = xyz_camXs - shift_am
        # xyz_max = torch.max(xyz_camXs)
        scene_centroid_x_noise = 0.0 #np.random.normal(0, 0.2) #+ offset_x
        scene_centroid_y_noise = 0.0 #np.random.normal(0, 0.2) #+ offset_y
        scene_centroid_z_noise = 0.0 #np.random.normal(0, 0.2) #+ offset_z
        if "depth_at_center" in feed.keys():
            depth_at_center, center_to_boundary = (feed["depth_at_center"], feed["center_to_boundary"])
            scene_centroid_B = torch.tensor([0.0+scene_centroid_x_noise, 0.0+scene_centroid_y_noise, depth_at_center+scene_centroid_z_noise]).unsqueeze(0).repeat(self.B,1).cuda()
            center_to_boundary = center_to_boundary #torch.max(center_to_boundary)
            bounds_B = torch.tensor([-center_to_boundary, center_to_boundary, -center_to_boundary, center_to_boundary, -center_to_boundary, center_to_boundary]).cuda()
        else:
            xyz_max = torch.clip(torch.max(xyz_maxs, dim=1)[0], max=20.0) / 2.
            scene_centroid_B = torch.tensor([np.zeros(1)+scene_centroid_x_noise, np.zeros(1)+scene_centroid_y_noise, xyz_max+scene_centroid_z_noise]).cuda().squeeze(0).reshape(1,3)
            # scene_centroid_B = torch.tensor([np.zeros(1)+scene_centroid_x_noise, np.zeros(1)+scene_centroid_y_noise, xyz_max+scene_centroid_z_noise]).cuda().squeeze(0).reshape(1,3)
            bounds_B = torch.stack([-xyz_max, xyz_max, -xyz_max, xyz_max, -xyz_max, xyz_max]).cuda()

        self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, "test", scene_centroid_B, bounds=bounds_B, assert_cube=True)
        # if False: #'markdata_enc3d_emb3d_07_repcarl'==set_name:
        #     # xyz_camXs = utils.geom.depth2pointcloud(depth_cam, pix_T_camX).float()

        #     xyz_maxs = torch.max(self.xyz_camXs, dim=2)[0]
        #     xyz_mins = torch.min(self.xyz_camXs, dim=1)[0]

        #     # print(xyz_maxs)
        #     # print(xyz_mins)

        #     xyz_max = torch.max(xyz_maxs, dim=1)[0]/2.
        #     self.scene_centroid = torch.tensor([0.0, 0.0, xyz_max]).unsqueeze(0).repeat(self.B,1).cuda()
        #     bounds = torch.tensor([-xyz_max, xyz_max, -xyz_max, xyz_max, -xyz_max, xyz_max]).cuda()
        #     self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, set_name, scene_centroid, bounds=bounds, assert_cube=True)

            

        # else:
        #     pass

            # depth_at_center, center_to_boundary = (feed["depth_at_center"], feed["center_to_boundary"])
            # scene_centroid_B = torch.tensor([0.0+scene_centroid_x_noise, 0.0+scene_centroid_y_noise, depth_at_center[b]+scene_centroid_z_noise]).unsqueeze(0).repeat(self.B,1).cuda()
            # center_to_boundary = center_to_boundary[b] #torch.max(center_to_boundary)
            # bounds_B = torch.tensor([-center_to_boundary, center_to_boundary, -center_to_boundary, center_to_boundary, -center_to_boundary, center_to_boundary]).cuda()
            
            # xyz_maxs = torch.max(self.xyz_camXs[:,1], dim=1)[0]
            # xyz_mins = torch.min(self.xyz_camXs[:,1], dim=1)[0]

            # # print(xyz_maxs)
            # # print(xyz_mins)

            # # shift_am = torch.tensor([(xyz_maxs[0][0] - torch.abs(xyz_mins[0][0]))/2., 0., 0.]).cuda().unsqueeze(0).unsqueeze(0)
            # # xyz_camXs = xyz_camXs - shift_am
            # # xyz_max = torch.max(xyz_camXs)
            # scene_centroid_x_noise = np.random.normal(0, 0.2)
            # scene_centroid_y_noise = np.random.normal(0, 0.2)
            # scene_centroid_z_noise = np.random.normal(0, 0.2)
            # xyz_max = torch.max(xyz_maxs, dim=1)[0]/2.
            # if hyp.clip_bounds is not None:
            #     xyz_max = torch.clip(xyz_max, min=-hyp.clip_bounds, max=hyp.clip_bounds)
            # self.scene_centroid = torch.tensor([0.0+scene_centroid_x_noise, 0.0+scene_centroid_y_noise, xyz_max+scene_centroid_z_noise]).unsqueeze(0).repeat(self.B,1).cuda()
            # bounds = torch.tensor([-xyz_max, xyz_max, -xyz_max, xyz_max, -xyz_max, xyz_max]).cuda()
            # # if hyp.clip_bounds is not None:
            # #     bounds = torch.clip(bounds, min=-hyp.clip_bounds, max=hyp.clip_bounds)
            # print(bounds)
            # print(torch.median(xyz_maxs, dim=1))


            # depth_at_center, center_to_boundary = (feed["depth_at_center"], feed["center_to_boundary"])
            # # scene_centroid_x_noise = np.random.normal(0, 0.2)
            # # scene_centroid_y_noise = np.random.normal(0, 0.2)
            # # scene_centroid_z_noise = np.random.normal(0, 0.2)
            # self.scene_centroid = torch.tensor([0.0, 0.0, depth_at_center]).unsqueeze(0).repeat(self.B,1).cuda()
            # bounds = torch.tensor([-center_to_boundary, center_to_boundary, -center_to_boundary, center_to_boundary, -center_to_boundary, center_to_boundary]).cuda()
            # self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, 'extract', self.scene_centroid, bounds=bounds, assert_cube=True)

            # occs = []
            # occs_X0 = []
            # rgbs = []
            # for s in range(self.xyz_camXs.shape[1]):
            #     occXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs[:,s:s+1]), self.Z, self.Y, self.X))
            #     occX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s[:,s:s+1]), self.Z, self.Y, self.X))
            #     occXs = occXs.squeeze()
            #     occXs_max = torch.max(occXs, 1).values
            #     occs.append(occXs_max)
            #     rgbs.append(self.rgb_camXs[:,s:s+1].squeeze())
            #     occX0s = occX0s.squeeze()
            #     occX0s_max = torch.max(occX0s, 1).values
            #     occs_X0.append(occX0s_max)
            #     if s%10==0:
            #         occs = torch.cat(occs, dim=1)
            #         plt.figure()
            #         plt.imshow(occs.cpu().numpy())
            #         plt.savefig('images/test.png')
            #         rgbs = torch.cat(rgbs, dim=2)
            #         plt.figure()
            #         plt.imshow(rgbs.permute(1,2,0).cpu().numpy()+0.5)
            #         plt.savefig('images/test2.png')
            #         occs_X0 = torch.cat(occs_X0, dim=1)
            #         plt.figure()
            #         plt.imshow(occs_X0.cpu().numpy())
            #         plt.savefig('images/test3.png')
            #         st()
            #         occs = []
            #         rgbs = []
            #         occs_X0 = []
                


                    
        # else:
        #     if feed['set_name']=='test':
        #         # center on an object, so that it does not fall out of bounds
        #         scene_centroid = utils.geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0]
        #     else:
        #         # center randomly 
        #         scene_centroid_x_noise = np.random.normal(0, 0.2)
        #         scene_centroid_y_noise = np.random.normal(0, 0.2)
        #         scene_centroid_z_noise = np.random.normal(0, 0.2)
        #         sc_noise = np.array([scene_centroid_x_noise, scene_centroid_y_noise, scene_centroid_z_noise])
        #         # scene_centroid = torch.median(self.xyz_camXs[0,0], dim=0)[0] + torch.from_numpy(sc_noise).float().cuda()
        #         # scene_centroid = np.array([scene_centroid_x,
        #         #                            scene_centroid_y,
        #         #                            scene_centroid_z]).reshape([1, 3])
        #         # scene_centroid = scene_centroid.float().cuda().reshape([1, 3])
        #         # print(sc_noise)
        #         scene_centroid = torch.from_numpy(sc_noise).float().cuda().reshape([1, 3])

        #     self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, feed['set_name'], scene_centroid=scene_centroid, assert_cube=True)

        # depth_camXs_, valid_camXs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        # self.summ_writer.summ_oneds(f'inputs/depth', __u(depth_camXs_).unbind(1))
        # dense_xyz_camXs_ = utils.geom.depth2pointcloud(depth_camXs_, __p(self.pix_T_cams))
        # dense_xyz_camRs_ = utils.geom.apply_4x4(__p(self.camRs_T_camXs), dense_xyz_camXs_)
        # inbound_camXs_ = self.vox_util.get_inbounds(dense_xyz_camRs_, self.Z, self.Y, self.X).float()
        # inbound_camXs_ = torch.reshape(inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        # # depth_camXs = __u(depth_camXs_)
        # self.valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)

    def get_processed_data(self, filename):
        # data = np.load(filename, allow_pickle=True).item()

        data = np.load(os.path.join(self.data_root, filename), allow_pickle=True).item()
        
        if self.dataset_name is not None:
            if self.dataset_name == "markdata":
                data = AttrDict(data)
                # blender camera is flipped, i.e., z is pointing outward
                data.camR_T_camXs[:,:,1] = data.camR_T_camXs[:,:,1] * (-1)
                data.camR_T_camXs[:,:,2] = data.camR_T_camXs[:,:,2] * (-1)
                camR_T_camXs = []
                for origin_T_camX in data.camR_T_camXs:
                    camR_T_camX = np.dot(self.adam_T_origin, origin_T_camX)
                    camR_T_camXs.append(camR_T_camX)
                data.camR_T_camXs  = np.stack(camR_T_camXs, axis=0)


        """
        rgb_camXs: nviews x 128 x 128 x 3
        depth_camXs: nviews x 128 x 128
        pix_T_cams: nviews x 3 x3
        camR_T_camXs: nviews x 4 x 4
        bbox_camR: 8x3
        cluster_id: string
        """
        if data.rgb_camXs.shape[1] == 256:
            data.rgb_camXs = data.rgb_camXs[:, ::2, ::2, :]
            data.depth_camXs = data.depth_camXs[:, ::2, ::2]
            data.pix_T_cams[:,0,0] = data.pix_T_cams[:,0,0] * 0.5
            data.pix_T_cams[:,1,1] = data.pix_T_cams[:,1,1] * 0.5
            data.pix_T_cams[:,0:2,2] = data.pix_T_cams[:,:2,2] * 0.5


        if self.dataset_name == "markdata":
            rgb_camXs = data.rgb_camXs[:]
            depth_camXs = data.depth_camXs[:]
            # frame 1 is shoot from ref cam

            

            camR_T_camXs = data.camR_T_camXs[:]
            pix_T_cams = data.pix_T_cams[:]
            origin_T_camR = np.eye(4, dtype=np.float32) #np.linalg.inv(data.camR_T_origin)
            origin_T_camRefs = data.camR_T_camXs[:1]
            rgb_camRefs = data.rgb_camXs[:1]
            depth_camRef = data.depth_camXs[0]
            depth_at_center = np.median(depth_camRef)

            H, W = depth_camRef.shape
            h = (H-1)/2
            boundary_pixel = []
            left_pixel = np.zeros((3))
            left_pixel[1] = h#int(H/2)h
            left_pixel[2] = depth_at_center

            right_pixel = np.zeros((3))
            right_pixel[0] = int(W)
            right_pixel[1] = h #int(H/2)
            right_pixel[2] = depth_at_center
            boundary_pixel.append(left_pixel)
            boundary_pixel.append(right_pixel)

            points = np.stack(boundary_pixel, axis=0)
            bd_pts = utils.geom.unproject_pts(points, data.pix_T_cams[0], origin_T_camRefs[0],
                                clip_radius=1000.0) # no limit

            center = np.mean(bd_pts, axis=0)
            center_to_boundary = np.linalg.norm(center - bd_pts, axis=1)[0] * 2.0


        else:
            rgb_camXs = data.rgb_camXs[:-1]
            depth_camXs = data.depth_camXs[:-1]
            # last image is shoot from ref cam

            camR_T_camXs = data.camR_T_camXs[:-1]
            pix_T_cams = data.pix_T_cams[:-1]
            origin_T_camR = np.eye(4, dtype=np.float32) #np.linalg.inv(data.camR_T_origin)
            origin_T_camRefs = data.camR_T_camXs[-1:]
            rgb_camRefs = data.rgb_camXs[-1:]


        # now ref_frame_bbox should also be added to the mix
        if self.dataset_name == "markdata":
            bbox_in_ref_cam = np.zeros((8,3), dtype=np.float32)
        else:
            bbox_in_ref_cam = data.bbox_camR
        

        if "cluster_id" in data:
            #if "cluster_id" in data
            cluster_id = data.cluster_id
        else:
            cluster_id = 0

        # below I am breaking 51 images into S equal parts randomly
        # sample 10 times

        #print("here1", getpid(), record_id, "/", len(records))

        rgbs = np.transpose(rgb_camXs, [0,3,1,2])
        depths = depth_camXs

        rgb_refs = np.transpose(rgb_camRefs, [0,3,1,2])

        origin_T_camXs = camR_T_camXs
        pix_T_cam = pix_T_cams
        num_views = len(rgbs)
        # this is actually identities
        origin_T_camRs = np.reshape(origin_T_camR, [1, 4, 4])
        origin_T_camRs = np.tile(origin_T_camRs, [num_views , 1, 1])

        rgbs = torch.from_numpy(rgbs).float()
        rgb_refs = torch.from_numpy(rgb_refs).float()
        depths = torch.from_numpy(depths).float().unsqueeze(1)
        pix_T_cam = torch.from_numpy(pix_T_cam).float()
        pix_T_cam_ = torch.eye(4).unsqueeze(0).repeat(pix_T_cam.shape[0], 1, 1).float()
        pix_T_cam_[:,:3,:3] = pix_T_cam
        pix_T_cam = pix_T_cam_
        origin_T_camXs = torch.from_numpy(origin_T_camXs).float()
        origin_T_camRs = torch.from_numpy(origin_T_camRs).float()

        if "success_rates_over_class" in data.keys():
            success_rates = torch.from_numpy(np.array(data.success_rates_over_class)).float()
        else:
            success_rates = torch.from_numpy(np.zeros((30), np.float32))

        #print("here2", getpid(), record_id, "/", len(records))
        xyz_camXs = utils.geom.depth2pointcloud(depths, pix_T_cam, cuda=False)

        #print("here3", getpid(), record_id, "/", len(records))

        camR_T_camX = origin_T_camXs
        xyz_camRs = utils.geom.apply_4x4(camR_T_camX, xyz_camXs)

        if len(bbox_in_ref_cam.shape) == 2:
            bbox_in_ref_cam = np.expand_dims(bbox_in_ref_cam, axis=0)
            cluster_id = [cluster_id]
        bbox_in_ref_cam = torch.from_numpy(bbox_in_ref_cam).float()

        #print("here4", getpid(), record_id, "/", len(records))

        d = dict()
        rgbs = rgbs / 255.
        rgb_refs = rgb_refs / 255.
        rgbs = rgbs - 0.5
        rgb_refs = rgb_refs - 0.5

        d['rgb_camXs'] = rgbs # numseq x 3 x 128 x 12
        d['rgb_camRs'] = rgb_refs # 1 x 3 x 128 x 128
        d['depth_camXs'] = depths # numseq x 1 x 128 x 128
        d['pix_T_cams'] = pix_T_cam # numseq x 3 x 3
        d['origin_T_camXs'] = origin_T_camXs # numseq x 4 x 4
        d['origin_T_camRs'] = origin_T_camRs # numseq x 4 x 4
        d['origin_T_camRefs'] = origin_T_camRefs

        d['xyz_camXs'] = xyz_camXs # numseq x V x 3
        d['xyz_camRs'] = xyz_camRs # numseq x V x 3
        d['bbox_in_ref_cam'] = bbox_in_ref_cam # Nobjs x 8 x3
        d['cluster_id'] = cluster_id # string
        d['record'] = filename # string, record_path
        d['success_rates'] = success_rates

        if self.dataset_name == "markdata":
            d["depth_at_center"] = depth_at_center
            d["center_to_boundary"] = center_to_boundary
        
        item_names = [
                'rgb_camXs',
                'depth_camXs',
                'pix_T_cams',
                'origin_T_camXs',
                'origin_T_camRs',
                'xyz_camXs',
                'xyz_camRs',
                ]

        for item_name in item_names:
            assert item_name in list(d.keys())
            #print("item_name", item_name)
            #print(chosen_record[item_name].shape)
            #print("===", indices)
            d[item_name] = d[item_name].unsqueeze(0).cuda()

        return d

if __name__ == '__main__':
    model = CARLA_MOC(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    model.initialize_model()