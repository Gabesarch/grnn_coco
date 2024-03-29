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
import torchvision
# from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from matplotlib import cm
# import seaborn as sn

from pycocotools.coco import COCO

# from backend import saverloader, inputs


import sys
# sys.path.append("XTConsistency")
# from modules.unet import UNet, UNetReshade

np.set_printoptions(precision=2)
np.random.seed(0)

import ipdb
st = ipdb.set_trace

# import dino.utils


sys.path.append("dino")
import dino_utils as utils_dino
import vision_transformer as vits
import colorsys
import random
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from vision_transformer import DINOHead

do_feat3d = True

XTC_init = '/home/gsarch/repo/pytorch_disco/saved_checkpoints/XTC_checkpoints/rgb2depth_consistency_wimagenet.pth'
coco_images_path = '/lab_data/tarrlab/common/datasets/NSD_images'

# set_name = 'dino'
# set_name = 'dino_all_layers'
# set_name = 'dino_all_layers_init'
# set_name = 'test02_inittest_0epoch'
set_name = 'dino_replica_carla'
set_name = 'dino_replica_carla_300epoch'

set_name = 'dino_mult_rep_carl07'
set_name = 'dino_mult_rep_carl_sameview07'

set_name = 'dino_vitsmall_omnidata'
set_name = 'dino_vitsmall_multiview_omnidata'

# checkpoint_path = "/home/gsarch/repo/grnn_coco/pytorch_disco/checkpoints/64_m144x144x144_1e-4_ns_dino_mult_rep_carl07/model-40000.pth"
# checkpoint_path = "/home/gsarch/repo/grnn_coco/pytorch_disco/checkpoints/64_m144x144x144_1e-4_ns_dino_mult_rep_carl_sameview07/model-40000.pth"
# checkpoint_path = "/lab_data/tarrlab/gsarch/omnidata/dino_checkpoints/vit/dino_vitsmall_100epochs/checkpoint.pth"
checkpoint_path = "/lab_data/tarrlab/gsarch/omnidata/dino_checkpoints/vit/dino_vitsmall_multiview_100epochs/checkpoint.pth"


use_dino_load = True

pretrain = True

checkpoint_dir='checkpoints/' + set_name
log_dir='logs_grnn_coco'

save_feats = True
plot_classes = False
only_process_stim_ids = False
log_freq = 1000 # frequency for logging tensorboard
subj = 1 # subject number - supports: 1, 2, 7 - only for when only_process_stim_ids=True


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

class DINO(nn.Module):
    def __init__(self):
        super(DINO, self).__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # # build model
        # self.model = vits.__dict__['vit_small'](patch_size=16) #, num_classes=0)
        # for p in self.model.parameters():
        #     p.requires_grad = False
        # self.model.eval()
        # self.model.to(self.device)
        # if pretrain:
        #     url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth" # model used for visualizations in DINO paper
        #     state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            # self.model.load_state_dict(state_dict, strict=True)
            
        # path = '/user_data/gsarch/dino_output/checkpoint0100.pth'
        # path = '/user_data/gsarch/dino_output_300epoch/checkpoint0300.pth'
        path = checkpoint_path

        print("PATH IS:", path)
        arch = "vit_small"
        patch_size = 16

        if "vit" in arch:
            model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
            print(f"Model {arch} {patch_size}x{patch_size} built.")
        elif "xcit" in arch:
            model = torch.hub.load('facebookresearch/xcit', arch, num_classes=0)
        elif arch in torchvision_models.__dict__.keys():
            model = torchvision_models.__dict__[arch](num_classes=0)
            model.fc = nn.Identity()
        else:
            print(f"Architecture {arch} non supported")
            sys.exit(1)
        model.cuda()

        # to_restore = {"epoch": 0}
        # dino.utils.restart_from_checkpoint(
        #     path,
        #     run_variables=to_restore,
        #     teacher=model,
        # )
        # st()
        # model = dino.utils.MultiCropWrapper(
        #     model,
        #     DINOHead(embed_dim, 65536, False),
        # )

        if use_dino_load:
            utils_dino.load_pretrained_weights(model, path, "teacher", arch, patch_size)
            # to_restore = {"epoch": 0}
            # utils_dino.restart_from_checkpoint(
            #     path,
            #     run_variables=to_restore,
            #     # student=student,
            #     teacher=model,
            #     # optimizer=optimizer,
            #     # fp16_scaler=fp16_scaler,
            #     # dino_loss=dino_loss,
            # )
        else:
            state_dict = torch.load(path)
            state_dict = state_dict["model_state_dict"]
            state_dict_ = {}
            for k, v in state_dict.items():
                if "teacher." in k:
                    state_dict_[k] = state_dict[k]
            state_dict = state_dict_
            state_dict = {k.replace("teacher.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            state_dict_ = {}
            for k, v in state_dict.items():
                if "head." not in k:
                    state_dict_[k] = state_dict[k]
            state_dict = state_dict_

            msg = model.load_state_dict(state_dict, strict=True)
            print("Loaded with message:", msg)

        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        self.model = model

        self.transform = transforms.Compose([
            # transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def forward(self, x):
        """Forward pass.
        Args:
            x (PIL): input data (image)
        Returns:
            tensor: depth
        """

        x = self.transform(x).unsqueeze(0)
        attentions = self.model.get_last_selfattention(x.to(self.device))
        # y = self.model.get_intermediate_layers(x.to(self.device), n=1)
        # y = self.model(x.to(self.device)) #).get_intermediate_layers(x.to(self.device), n=1)
        output = self.model.get_intermediate_layers(x.to(self.device), n=12)
        out0 = output[0][:, 0]
        out1 = output[1][:, 0]
        out2 = output[2][:, 0]
        out3 = output[3][:, 0]
        out4 = output[4][:, 0]
        out5 = output[5][:, 0]
        out6 = output[6][:, 0]
        out7 = output[7][:, 0]
        out8 = output[8][:, 0]
        out9 = output[9][:, 0]
        out10 = output[10][:, 0]
        out11 = output[11][:, 0]

        self.model(x.to(self.device))

        return out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, attentions

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
        self.dino = DINO() #torch.nn.Sequential(*(list(midas.pretrained.children())+list(midas.scratch.children()))[:-1])
        self.dino.cuda()
        self.dino.eval()
        # print(self.dino)

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
                            transforms.ToTensor(),
                            ])
        dataset = ImageFolderWithPaths(coco_images_path, transform=data_loader_transform) # our custom dataset
        
        # self.dataloader = torch.utils_DataLoader(dataset, batch_size=32, shuffle=False)
        # dataset = datasets.ImageFolder(coco_images_path, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.B, shuffle=False)

        self.writer = SummaryWriter(log_dir + '/' + set_name, max_queue=10, flush_secs=1000)

        self.toPIL = transforms.ToPILImage()

    
    def run_extract(self):
        
        if only_process_stim_ids:
            feats = np.zeros((10000, 147456))
            file_order = np.zeros(10000)
        else:
            out0_feats = np.zeros((73000, 384)).astype(np.float32)
            out1_feats = np.zeros((73000, 384)).astype(np.float32)
            out2_feats = np.zeros((73000, 384)).astype(np.float32)
            out3_feats = np.zeros((73000, 384)).astype(np.float32)
            out4_feats = np.zeros((73000, 384)).astype(np.float32)
            out5_feats = np.zeros((73000, 384)).astype(np.float32)
            out6_feats = np.zeros((73000, 384)).astype(np.float32)
            out7_feats = np.zeros((73000, 384)).astype(np.float32)
            out8_feats = np.zeros((73000, 384)).astype(np.float32)
            out9_feats = np.zeros((73000, 384)).astype(np.float32)
            out10_feats = np.zeros((73000, 384)).astype(np.float32)
            out11_feats = np.zeros((73000, 384)).astype(np.float32)
            attention = np.zeros((73000, 4056)).astype(np.float32)
            file_order = np.zeros(73000)
        cat_names = []
        supcat_names = []
        cat_ids = []
        idx = 0
        # for images, file_ids in tqdm(enumerate(self.dataloader)):
        for images, _, file_ids in tqdm(self.dataloader):

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
            images = images.squeeze(0)
            images_pil = self.toPIL(images).convert('RGB')

            if self.summ_writer.save_this:
                rgb_camX = images.cuda().float().unsqueeze(0)
                rgb_camX_norm = rgb_camX - 0.5
                self.summ_writer.summ_rgb('inputs/rgb', rgb_camX_norm)
                del rgb_camX, rgb_camX_norm

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
            out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, attention_ = self.dino(images_pil)

            nh = attention_.shape[1] # number of head

            # we keep only the output patch attention
            attention_ = attention_[0, :, 0, 1:]#.reshape(nh, -1)

            if save_feats:
                attention_ = attention_.reshape(self.B, -1)
                out0_feats[idx] = out0.squeeze().detach().cpu().numpy() 
                out1_feats[idx] = out1.squeeze().detach().cpu().numpy() 
                out2_feats[idx] = out2.squeeze().detach().cpu().numpy() 
                out3_feats[idx] = out3.squeeze().detach().cpu().numpy() 
                out4_feats[idx] = out4.squeeze().detach().cpu().numpy() 
                out5_feats[idx] = out5.squeeze().detach().cpu().numpy() 
                out6_feats[idx] = out6.squeeze().detach().cpu().numpy() 
                out7_feats[idx] = out7.squeeze().detach().cpu().numpy() 
                out8_feats[idx] = out8.squeeze().detach().cpu().numpy() 
                out9_feats[idx] = out9.squeeze().detach().cpu().numpy() 
                out10_feats[idx] = out10.squeeze().detach().cpu().numpy() 
                out11_feats[idx] = out11.squeeze().detach().cpu().numpy() 
                attention[idx] = attention_.detach().cpu().numpy() 
                file_order[idx] = file_ids.detach().cpu().numpy() 

            if False: #self.summ_writer.save_this:
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
            np.save(f'{output_dir}/out0_feats.npy', out0_feats)
            np.save(f'{output_dir}/out1_feats.npy', out1_feats)
            np.save(f'{output_dir}/out2_feats.npy', out2_feats)
            np.save(f'{output_dir}/out3_feats.npy', out3_feats)
            np.save(f'{output_dir}/out4_feats.npy', out4_feats)
            np.save(f'{output_dir}/out5_feats.npy', out5_feats)
            np.save(f'{output_dir}/out6_feats.npy', out6_feats)
            np.save(f'{output_dir}/out7_feats.npy', out7_feats)
            np.save(f'{output_dir}/out8_feats.npy', out8_feats)
            np.save(f'{output_dir}/out9_feats.npy', out9_feats)
            np.save(f'{output_dir}/out10_feats.npy', out10_feats)
            np.save(f'{output_dir}/out11_feats.npy', out11_feats)
            np.save(f'{output_dir}/attention.npy', attention)
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