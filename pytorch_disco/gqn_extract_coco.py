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

import sys
sys.path.append("torch-gqn")
from model import GQN
from torch.distributions import Normal

from torchvision.utils import make_grid, save_image

from torchvision.transforms import ToTensor, Resize, ToPILImage, CenterCrop


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

set_name = 'test00' # take output of entire network
set_name = 'gqn_pool' # all layers
# set_name = 'gqn_pool_vis' # visualize gqn output
# set_name = 'gqn_pool_vis_test2' # visualize gqn output
set_name = 'gqn_pool_v2' # all layers
set_name = 'gqn_test03' # all layers
set_name = 'gqn_pool_v3' # all layers
# set_name = 'gqn_rel_test02' # all layers
set_name = 'gqn_pool_rel' # all layers



checkpoint_dir='checkpoints/' + set_name
log_dir='logs_carla_moc'

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

import argparse
parser = argparse.ArgumentParser(description='Generative Query Network Implementation')
parser.add_argument('--gradient_steps', type=int, default=2*10**6, help='number of gradient steps to run (default: 2 million)')
parser.add_argument('--batch_size', type=int, default=36, help='size of batch (default: 36)')
parser.add_argument('--dataset', type=str, default='Shepard-Metzler', help='dataset (dafault: Shepard-Mtzler)')
parser.add_argument('--train_data_dir', type=str, help='location of training data', \
                    default="/workspace/dataset/shepard_metzler_7_parts-torch/train")
parser.add_argument('--test_data_dir', type=str, help='location of test data', \
                    default="/workspace/dataset/shepard_metzler_7_parts-torch/test")
parser.add_argument('--root_log_dir', type=str, help='root location of log', default='/workspace/logs')
parser.add_argument('--log_dir', type=str, help='log directory (default: GQN)', default='GQN')
parser.add_argument('--log_interval', type=int, help='interval number of steps for logging', default=100)
parser.add_argument('--save_interval', type=int, help='interval number of steps for saveing models', default=10000)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
# parser.add_argument('--representation', type=str, help='representation network (default: pool)', default='pool')
parser.add_argument('--layers', type=int, help='number of generative layers (default: 12)', default=12)
parser.add_argument('--shared_core', type=bool, \
                    help='whether to share the weights of the cores across generation steps (default: False)', \
                    default=False)
parser.add_argument('--seed', type=int, help='random seed (default: None)', default=None)
parser.add_argument("--mode","--m", default="moc", help="experiment mode")
parser.add_argument("--exp_name","--en", default="trainer_basic", help="execute expriment name defined in config")
parser.add_argument("--run_name","--rn", default="1", help="run name")
args = parser.parse_args()

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

class GQN_MODEL(nn.Module):
    def __init__(self):
        super(GQN_MODEL, self).__init__()

        # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        # model_type = "MiDaS"  

        # self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=pretrained).eval()
        # checkpoint = "/home/gsarch/repo/grnn_coco/pytorch_disco/checkpoints/01_m128x64x128_1e-4_carla_and_replica_train_ns_gqn_pool02/model-200000.pth"
        # checkpoint = "/home/gsarch/repo/grnn_coco/pytorch_disco/checkpoints/01_m128x64x128_1e-4_carla_and_replica_train_ns_gqn_pool_rel00/model-175000.pth"

        if True:
            L = args.layers
            self.gqn = GQN(representation=hyp.gqn_representation, L=L, shared_core=args.shared_core).cuda()

        pretrain = True
        if pretrain:
            # inits = {"gqn": "01_m128x64x128_1e-4_carla_and_replica_train_ns_gqn_pool02",}
            inits = {"gqn": "01_m128x64x128_1e-4_carla_and_replica_train_ns_gqn_pool_rel00",} # relative egomotion
            print(inits)
            for part, init in list(inits.items()):
                # st()
                print(part, init)
                if init:
                    if part == 'gqn':  
                        model_part = self.gqn
                    else:
                        assert(False)
                    iter = saverloader.load_part(model_part, part, init)
                    if iter:
                        print("loaded %s at iter %d" % (init, iter))
                    else:
                        print("could not find a checkpoint for %s" % init)

        for p in self.gqn.parameters():
            p.requires_grad = False
        self.gqn.eval()

        # yaw = 0.0
        # pitch = 0.0
        # trans = 
        # view_vector = [trans, torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1), torch.cos(pitch).unsqueeze(1), torch.sin(pitch).unsqueeze(1)]
        # v_hat = torch.cat(view_vector, dim=-1).reshape(self.B, self.S, 7)

        self.v = torch.zeros((1,1,7))


    def forward(self, x, v=None, v_q=None):
        """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """

        feats = {}

        B, M, *_ = x.size()

        if v is None:
            assert(False)
            # v = torch.zeros((B,1,7)).cuda()
            # v_q = torch.zeros((B,1,7)).cuda()
        
        # Scene encoder
        if self.gqn.representation=='tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            skip_in1, skip_out1, r1, r2, skip_out2, r3, r4, r5, r6 = self.gqn.phi.forward_feats(x[:, k], v[:, k]) # skip_in1, skip_out1, r1, r2, skip_out2, r3, r4, r5, r6
            r += r6

        feats['skip_in1'] = skip_in1
        feats['skip_out1'] = skip_out1
        feats['r1'] = r1
        feats['r2'] = r2
        feats['skip_out2'] = skip_out2
        feats['r3'] = r3
        feats['r4'] = r4
        feats['r5'] = r5
        feats['r6'] = r6

        # Initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # feats = []
        
        for l in range(self.gqn.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.gqn.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Prior sample
            z = pi.sample()
            
            # State update
            if self.gqn.shared_core:
                c_g, h_g, u = self.gqn.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.gqn.generation_core[l](v_q, r, c_g, h_g, u, z)

            dict_label = 'l' + str(l)
            feats[dict_label] = h_g
            
        # Image sample
        mu = self.gqn.eta_g(u)

        feats['mu'] = mu

        return feats

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
        self.gqn = GQN_MODEL() #torch.nn.Sequential(*(list(midas.pretrained.children())+list(midas.scratch.children()))[:-1])
        self.gqn.cuda()
        for p in self.gqn.parameters():
            p.requires_grad = False
        self.gqn.eval()
        # print(self.midas)

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
        self.pix_T_camX = torch.from_numpy(self.pix_T_camX).cuda().unsqueeze(0).repeat(self.B,1,1).float()

        data_loader_transform = transforms.Compose([
                            transforms.ToTensor()])
        dataset = ImageFolderWithPaths(coco_images_path, transform=data_loader_transform) # our custom dataset
        
        # self.dataloader = torch.utils_DataLoader(dataset, batch_size=32, shuffle=False)
        # dataset = datasets.ImageFolder(coco_images_path, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.B, shuffle=False)

        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X

        # self.Z = 128
        # self.Y = 128
        # self.X = 128
        bounds = torch.tensor([-12.0, 12.0, -12.0, 12.0, -12.0, 12.0]).cuda()
        self.scene_centroid = torch.tensor([0.0, 0.0, 10.0]).unsqueeze(0).repeat(self.B,1).cuda()
        self.vox_util = utils_vox.Vox_util(self.Z, self.Y, self.X, set_name, scene_centroid=self.scene_centroid, assert_cube=True, bounds=bounds)

        self.writer = SummaryWriter(log_dir + '/' + set_name, max_queue=1, flush_secs=100)

        # self.avgpool3d = nn.AvgPool3d(2, stride=2)
        self.pool_len = pool_len
        self.pool2d = nn.AvgPool2d(self.pool_len, stride=self.pool_len)

        self.num_maxpool = num_maxpool

        # self.run_extract()

    # def get_qgn_egomotion(self, origin_T_camXs):
    #     rot,trans = utils.geom.split_rt(origin_T_camXs.reshape(self.B*self.S,4,4))
    #     yaw, pitch, roll = utils.geom.rotm2eul(rot)
    #     view_vector = [trans, torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1), torch.cos(pitch).unsqueeze(1), torch.sin(pitch).unsqueeze(1)]
    #     v_hat = torch.cat(view_vector, dim=-1).reshape(self.B, self.S, 7)
    #     return v_hat
    
    def run_extract(self):

        byte_to_tensor = lambda x: ToTensor()(Resize(64)(CenterCrop(min(x.shape[-1], x.shape[-2]))((ToPILImage()(x))))).cuda()

        if do_dim_red:
            ch = 15 # 15 pcs well captures 95% of variance
        else:
            ch = 256
        
        if only_process_stim_ids:
            feats = np.zeros((10000, ch,192/2,192/2), dtype=np.float32)
            file_order = np.zeros(10000)
        else:
            # skip_in1 = np.zeros((73000,256, 16, 16), dtype=np.float32)
            # skip_out1 = np.zeros((73000,256, 16, 16), dtype=np.float32)
            # r1 = np.zeros((73000,128, 32, 32), dtype=np.float32)
            # r2 = np.zeros((73000,256, 16, 16), dtype=np.float32)
            # skip_out2 = np.zeros((73000,256, 16, 16), dtype=np.float32)
            # r3 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # r4 = np.zeros((73000,256, 16, 16), dtype=np.float32)
            r5 = np.zeros((73000,256, 16, 16), dtype=np.float32)
            r6 = np.zeros((73000,256, 1, 1), dtype=np.float32)
            # l0 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # l1 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # l2 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # l3 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # l4 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # l5 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # l6 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # l7 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # l8 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # l9 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # l10 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            # l11 = np.zeros((73000,128, 16, 16), dtype=np.float32)
            file_order = np.zeros(73000)
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
            # rgb_camX_norm = rgb_camX - 0.5
            # self.summ_writer.summ_rgb('inputs/rgb', rgb_camX_norm)

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
            # rgb_camX = rgb_camX.permute(0,2,3,1).detach().cpu().numpy()
            # rgb_camX = rgb_camX.detach().cpu().numpy()
            # input_batch = []
            # for b in range(self.B):
            #     input_batch.append(self.transform(rgb_camX[b]).cuda())
            # input_batch = torch.cat(input_batch, dim=0)
            eps = 0.0 #1e-6
            yaw = torch.tensor([0.0 + eps])
            pitch = torch.tensor([0.0 + eps])
            trans = torch.tensor([eps, eps, eps]).unsqueeze(0)
            view_vector = [trans, torch.cos(yaw).unsqueeze(0), torch.sin(yaw).unsqueeze(0), torch.cos(pitch).unsqueeze(0), torch.sin(pitch).unsqueeze(0)]
            v_hat = torch.cat(view_vector, dim=-1).reshape(self.B, 1, 7)
            v_hat2 = torch.cat(view_vector, dim=-1).reshape(self.B, 1, 7)

            # # more realistic numbers - this seems to make a slight difference
            # v_hat = torch.tensor([-0.5616,  1.0793,  0.8348, -0.9962,  0.0872,  0.9434,  0.3315]).reshape(self.B, 1, 7)
            # v_hat2 = torch.tensor([-0.5616,  1.0793,  0.8348, -0.9962,  0.0872,  0.9434,  0.3315]).reshape(self.B, 1, 7)

            v = v_hat2.cuda() #torch.zeros((self.B,1,7)).cuda()
            v_q = v_hat.cuda() #torch.zeros((self.B,1,7)).cuda()
            
            input_batch = torch.stack([byte_to_tensor(frame) for frame in rgb_camX]).reshape(self.B,1,3,64,64)
            with torch.no_grad():
                layers_list = self.gqn(input_batch, v=v, v_q=v_q)

            # img = feats['mu']

            # x_q_rec_test = self.gqn.reconstruct(self.rgb_camXs[:,1:2]+0.5, v[:,1:2], self.qgn_v[:,0], self.rgb_camXs[:,0]+0.5)
            with torch.no_grad():
                x_q_hat_test = self.gqn.gqn.generate(input_batch, v, v_q)
            
            # self.summ_writer.summ_scalar('train_kl', kl_test.mean())
            self.summ_writer.summ_rgb('view/coco_ground_truth', input_batch[:,0]-0.5)
            # self.summ_writer.summ_rgb('view/train_reconstruction', x_q_rec_test-0.5)
            self.summ_writer.summ_rgb('view/coco_generation', x_q_hat_test-0.5)

            # layers_list = {'skip_in1':out3,'out2':out2, 'out1':out1, 'path_1':path_1, 'path_3':path_3, 'layer_4_rn':layer_4_rn, 'layer_2_rn':layer_2_rn, 'layer_4':layer_4, 'layer_2':layer_2}
            
            # depth_cam = (torch.max(depth_cam) - depth_cam) / 100.0

            # depth_cam = self.XTCmodel_depth(rgb_camXs)

            # # get depths in 0-100 range approx.
            # depth_cam = (depth_cam / 0.7) * 100
            
            # self.summ_writer.summ_depth('inputs/depth_map', feat_memX[0].squeeze().detach().cpu().numpy())

            if save_feats:
                # pooling

                # out3 = self.pool2d(out3)
                # out2 = self.pool2d(out2)
                # out1 = self.pool2d(out1)
                # path_1 = self.pool2d(path_1)

                do_pool = ['skip_in1']
                do_pool2 = []
                # do_reduce_chans = {'out2':4, 'out1':4, 'path_1':15, 'path_3':20, 'layer_4_rn':35, 'layer_2_rn':120, 'layer_4':100, 'layer_2':200}

                # layers_list = {'out2':out2, 'out1':out1, 'path_1':path_1, 'path_3':path_3, 'layer_4_rn':layer_4_rn, 'layer_2_rn':layer_2_rn, 'layer_4':layer_4, 'layer_2':layer_2}

                for key in list(layers_list.keys()):
                    if key in do_pool:
                        layers_list[key] = self.pool2d(layers_list[key])
                    if key in do_pool2: 
                        layers_list[key] = self.pool2d(layers_list[key])

                    # if key in list(do_reduce_chans.keys()):
                    #     ch = do_reduce_chans[key]
                    #     feat_memX = layers_list[key].squeeze()
                    #     feat_memX = feat_memX.permute(0,2,3,1)
                    #     b,h,w,c = feat_memX.shape
                    #     feat_memX = feat_memX.reshape(b, h*w, c).cpu().numpy()
                    #     feat_memX_red = np.zeros((b,ch,h,w), dtype=np.float32)
                    #     for b_i in range(self.B):
                    #         feat_memX_ = feat_memX[b_i]
                    #         pca = PCA(n_components=ch)
                    #         feat_memX_ = pca.fit_transform(feat_memX_)
                    #         feat_memX_= feat_memX_.reshape(h,w,ch).transpose((2,0,1))
                    #         feat_memX_red[b_i] = feat_memX_
                    #         # print(key, feat_memX_.shape)
                    #         # print(key, np.cumsum(pca.explained_variance_ratio_)[-1])
                    #     # feat_memX = feat_memX_red
                    #     layers_list[key] = feat_memX_red
                    #     del feat_memX_red
                    if torch.is_tensor(layers_list[key]):
                        layers_list[key] = layers_list[key].cpu().numpy()
                    # print(key, layers_list[key].reshape(self.B, -1).shape)
                    # print(key, layers_list[key].shape)
                        # feat_memX = torch.from_numpy(feat_memX).cuda()

                        # plt.plot(np.cumsum(pca.explained_variance_ratio_))
                        # plt.xlabel('number of components')
                        # plt.ylabel('cumulative explained variance')
                        # plt.yticks(np.arange(0.0,1.0,0.05))
                        # plt.savefig('images/variance_explained.png')
                        # st()                
    
                # 2d pool
                # if self.num_maxpool>0:
                #     for nm in range(self.num_maxpool):
                #         feat_memX = self.pool2d(feat_memX)

                # feats.append(feat_memX.detach().cpu().numpy())
                # file_order.append(file_ids.detach().cpu().numpy())
                # st()
                # feat_memX = feat_memX.reshape(self.B, -1)


                # skip_in1[idx:idx+self.B] = layers_list['skip_in1'].astype(np.float32)
                # skip_out1[idx:idx+self.B]= layers_list['skip_out1'].astype(np.float32)
                # r1[idx:idx+self.B]= layers_list['r1'].astype(np.float32)
                # r2[idx:idx+self.B]= layers_list['r2'].astype(np.float32)
                # skip_out2[idx:idx+self.B]= layers_list['skip_out2'].astype(np.float32)
                # r3[idx:idx+self.B]= layers_list['r3'].astype(np.float32)
                # r4[idx:idx+self.B]= layers_list['r4'].astype(np.float32)
                r5[idx:idx+self.B]= layers_list['r5'].astype(np.float32)
                r6[idx:idx+self.B]= layers_list['r6'].astype(np.float32)
                # l0[idx:idx+self.B]= layers_list['l0'].astype(np.float32)
                # l1[idx:idx+self.B]= layers_list['l1'].astype(np.float32)
                # l2[idx:idx+self.B]= layers_list['l2'].astype(np.float32)
                # l3[idx:idx+self.B]= layers_list['l3'].astype(np.float32)
                # l4[idx:idx+self.B]= layers_list['l4'].astype(np.float32)
                # l5[idx:idx+self.B]= layers_list['l5'].astype(np.float32)
                # l6[idx:idx+self.B]= layers_list['l6'].astype(np.float32)
                # l7[idx:idx+self.B]= layers_list['l7'].astype(np.float32)
                # l8[idx:idx+self.B]= layers_list['l8'].astype(np.float32)
                # l9[idx:idx+self.B]= layers_list['l9'].astype(np.float32)
                # l10[idx:idx+self.B]= layers_list['l10'].astype(np.float32)
                # l11[idx:idx+self.B]= layers_list['l11'].astype(np.float32)
                file_order[idx:idx+self.B] = file_ids.detach().cpu().numpy()        

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
            # np.save(f'{output_dir}/skip_in1.npy', skip_in1)
            # np.save(f'{output_dir}/skip_out1.npy', skip_out1)
            # np.save(f'{output_dir}/r1.npy', r1)
            # np.save(f'{output_dir}/r2.npy', r2)
            # np.save(f'{output_dir}/skip_out2.npy', skip_out2)
            # np.save(f'{output_dir}/r3.npy', r3)
            # np.save(f'{output_dir}/r4.npy', r4)
            np.save(f'{output_dir}/r5.npy', r5)
            np.save(f'{output_dir}/r6.npy', r6)
            # np.save(f'{output_dir}/l0.npy', l0)
            # np.save(f'{output_dir}/l1.npy', l1)
            # np.save(f'{output_dir}/l2.npy', l2)
            # np.save(f'{output_dir}/l3.npy', l3)
            # np.save(f'{output_dir}/l4.npy', l4)
            # np.save(f'{output_dir}/l5.npy', l5)
            # np.save(f'{output_dir}/l6.npy', l6)
            # np.save(f'{output_dir}/l7.npy', l7)
            # np.save(f'{output_dir}/l8.npy', l8)
            # np.save(f'{output_dir}/l9.npy', l9)
            # np.save(f'{output_dir}/l10.npy', l10)
            # np.save(f'{output_dir}/l11.npy', l11)
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

    


if __name__ == '__main__':
    model = CARLA_MOC(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    model.initialize_model()