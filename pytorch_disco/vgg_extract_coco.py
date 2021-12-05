import os
# script = """
# MODE="CARLA_MOC"
# export MODE
# """
# os.system("bash -c '%s'" % script)
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

from model_base import Model

import cv2

from tensorboardX import SummaryWriter
import torch.nn.functional as F

import utils_improc

from torchvision import transforms
from torchvision import datasets

import matplotlib.pyplot as plt
from matplotlib import cm
# import seaborn as sn

from pycocotools.coco import COCO

import torchvision
from torchvision import transforms
import PIL

from sklearn.decomposition import PCA


import sys
# sys.path.append("XTConsistency")
# from modules.unet import UNet, UNetReshade

np.set_printoptions(precision=2)
np.random.seed(0)

import ipdb
st = ipdb.set_trace

do_feat3d = True

coco_images_path = '/lab_data/tarrlab/common/datasets/NSD_images'

# set_name = 'grnn_tsne02'
# set_name = 'grnn_depth00'
set_name = 'vgg_tsne00'
set_name = 'vgg_tsne01'
set_name = 'vgg_tsne02' # new without pca fc features


set_name = 'vgg_all_layers' # new without pca fc features
set_name = 'vgg_all_layers_init'

pretrained = False


checkpoint_dir='checkpoints/' + set_name
log_dir='logs_vgg_coco'

output_dir = f'/lab_data/tarrlab/gsarch/encoding_model/{set_name}'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)



# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py

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

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        vgg16 = torchvision.models.vgg16(pretrained=pretrained).double().cuda()
        vgg16.eval()

        # vgg_layers = list(vgg16.features.children())
        # vgg_layers.extend([vgg16.avgpool])
        # vgg_layers.extend(list(vgg16.classifier.children()))
        # print(torch.nn.Sequential(*list(vgg16.features.children())))
        self.block1 = torch.nn.Sequential(*list(vgg16.features.children())[:5])
        self.block2 = torch.nn.Sequential(*list(vgg16.features.children())[5:10])
        self.block3 = torch.nn.Sequential(*list(vgg16.features.children())[10:17])
        self.block4 = torch.nn.Sequential(*list(vgg16.features.children())[17:24])
        self.block5 = torch.nn.Sequential(*list(vgg16.features.children())[24:31])
        # self.features = torch.nn.Sequential(*list(vgg16.features.children()))
        self.avgpool = vgg16.avgpool
        self.fc1 = torch.nn.Sequential(*list(vgg16.classifier.children())[:1])
        self.fc2 = torch.nn.Sequential(*list(vgg16.classifier.children())[1:4]) # first fc layer
        self.fc3 = torch.nn.Sequential(*list(vgg16.classifier.children())[4:]) # first fc layer

        self.vgg_mean = torch.from_numpy(np.array([0.485,0.456,0.406]).reshape(1,3,1,1)).cuda()
        self.vgg_std = torch.from_numpy(np.array([0.229,0.224,0.225]).reshape(1,3,1,1)).cuda()
        self.resize = transforms.Compose([
            # transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(256),
            # transforms.ToTensor(),
        ])
    
    def forward(self, rgb_camX, summ_writer=None):

        rgb_camX = F.interpolate(rgb_camX, size=256, mode='bilinear')
        rgb_camX = self.resize(rgb_camX)

        # if summ_writer is not None:
        #     rgb_camX_norm = rgb_camX - 0.5
        #     summ_writer.summ_rgb('inputs/rgb', rgb_camX_norm)

        rgb_camX = (rgb_camX - self.vgg_mean) / self.vgg_std

        b1 = self.block1(rgb_camX)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        b5 = self.block5(b4)
        x = self.avgpool(b5)
        x = torch.flatten(x, 1)
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)

        return b1, b2, b3, b4, b5, fc1_out, fc2_out, fc3_out
    
    # def forward(self, rgb_camX, summ_writer=None):

    #     rgb_camX = F.interpolate(rgb_camX, size=256, mode='bilinear')
    #     rgb_camX = self.resize(rgb_camX)

    #     if summ_writer is not None:
    #         rgb_camX_norm = rgb_camX - 0.5
    #         summ_writer.summ_rgb('inputs/rgb', rgb_camX_norm)

    #     rgb_camX = (rgb_camX - self.vgg_mean) / self.vgg_std

    #     x = self.features(rgb_camX)
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)

    #     return x



class CARLA_MOC(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = COCO_GRNN()

        # self.model.feat3dnet.eval()

        # self.start_iter = saverloader.load_weights(self.model, None)

        self.model.run_extract()
        # self.model.plot_tsne_only()
        # self.model.plot_depth()

class COCO_GRNN(nn.Module):
    def __init__(self):
        super(COCO_GRNN, self).__init__()
        
        # self.feat3dnet = Feat3dNet(in_dim=4, out_dim=32).cuda()
        # self.feat3dnet.eval()
        # self.set_requires_grad(self.feat3dnet, False)

        # path = 'saved_checkpoints/01_s2_m128x128x128_1e-4_F3_d32_O_c1_s.1_E3_n2_d16_c1_carla_and_replica_train_cr12/model-00070000.pth'
        # checkpoint = torch.load(path)
        # self.feat3dnet.load_state_dict(checkpoint['model_state_dict'])

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

        # self.COCO_cat_NSD = np.load(coco_images_path + '/NSD_cat_feat.npy')
        # self.COCO_supcat_NSD = np.load(coco_images_path + '/NSD_supcat_feat.npy')
        # self.coco_util_train = COCO(coco_images_path + '/annotations/instances_train2014.json')
        # self.coco_util_val = COCO(coco_images_path + '/annotations/instances_val2014.json')

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

        self.vgg = VGG().cuda().eval()
        print(self.vgg)

        self.W = 256
        self.H = 256
        # self.fov = 90
        # hfov = float(self.fov) * np.pi / 180.
        # self.pix_T_camX = np.array([
        #     [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
        #     [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
        #     [0., 0.,  1, 0],
        #     [0., 0., 0, 1]])
        # self.pix_T_camX[0,2] = self.W/2.
        # self.pix_T_camX[1,2] = self.H/2.
        

        self.B = 25
        # self.pix_T_camX = torch.from_numpy(self.pix_T_camX).cuda().unsqueeze(0).repeat(self.B,1,1).float()

        data_loader_transform = transforms.Compose([
                            transforms.ToTensor()])
        dataset = ImageFolderWithPaths(coco_images_path, transform=data_loader_transform) # our custom dataset
        # self.dataloader = torch.utils.DataLoader(dataset, batch_size=32, shuffle=False)
        # dataset = datasets.ImageFolder(coco_images_path, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.B, shuffle=False)

        # self.Z = 128
        # self.Y = 128
        # self.X = 128
        # bounds = torch.tensor([-12.0, 12.0, -12.0, 12.0, -12.0, 12.0]).cuda()
        # self.scene_centroid = torch.tensor([0.0, 0.0, 10.0]).unsqueeze(0).repeat(self.B,1).cuda()
        # self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, set_name, scene_centroid=self.scene_centroid, assert_cube=True, bounds=bounds)

        self.writer = SummaryWriter(log_dir + '/' + set_name, max_queue=10, flush_secs=1000)

        self.avgpool2d = nn.AvgPool2d(2, stride=2)



        # self.run_extract()
    
    def run_extract(self):
        
        feats = []
        file_order = []
        cat_names = []
        supcat_names = []
        cat_ids = []
        idx = 0
        b1_feats = np.zeros((73000, 64,32,32)).astype(np.float32)
        b2_feats = np.zeros((73000, 128,16,16)).astype(np.float32)
        b3_feats = np.zeros((73000, 256,16,16)).astype(np.float32)
        b4_feats = np.zeros((73000, 512,16,16)).astype(np.float32)
        b5_feats = np.zeros((73000, 512,8,8)).astype(np.float32)
        fc1_feats = np.zeros((73000, 4096)).astype(np.float32)
        fc2_feats = np.zeros((73000, 4096)).astype(np.float32)
        fc3_feats = np.zeros((73000, 1000)).astype(np.float32)
        # feats = np.zeros((73000, 256*32*32), dtype=np.float16)
        # file_order = np.zeros(73000)
        for images, _, file_ids in self.dataloader:

            print('Images processed: ', idx)

            self.summ_writer = utils_improc.Summ_writer(
                writer=self.writer,
                global_step=idx,
                set_name=set_name,
                log_freq=1000,
                fps=8,
                # just_gif=True,
            )

            rgb_camX = images.cuda().float()
            

            if False:
                plt.figure()
                rgb_camXs_np = rgb_camXs[0].permute(1,2,0).detach().cpu().numpy()
                plt.imshow(rgb_camXs_np)
                plt.savefig('images/test.png')


            # # get category name
            # for b in range(self.B):
            #     img_id = [int(file_ids[b].detach().cpu().numpy())]
            #     coco_util = self.coco_util_train
            #     annotation_ids = coco_util.getAnnIds(img_id)
            #     if not annotation_ids:
            #         coco_util = self.coco_util_val
            #         annotation_ids = coco_util.getAnnIds(img_id)
            #     annotations = coco_util.loadAnns(annotation_ids)

            #     best_area = 0
            #     entity_id = None
            #     entity = None
            #     for i in range(len(annotations)):
            #         if annotations[i]['area'] > best_area:
            #             entity_id = annotations[i]["category_id"]
            #             entity = coco_util.loadCats(entity_id)[0]["name"]
            #             super_cat = coco_util.loadCats(entity_id)[0]["supercategory"]
            #             best_area = annotations[i]['area']
            #     cat_names.append(entity)
            #     cat_ids.append(entity_id)
            #     supcat_names.append(super_cat)

            b1, b2, b3, b4, b5, fc1_out, fc2_out, fc3_out = self.vgg(rgb_camX, summ_writer=self.summ_writer)

            for avp in range(2):
                b1 = self.avgpool2d(b1)
            for avp in range(2):
                b2 = self.avgpool2d(b2)
            b3 = self.avgpool2d(b3)

            b1_feats[idx:idx+self.B] = b1.detach().cpu().numpy().astype(np.float32)
            b2_feats[idx:idx+self.B] = b2.detach().cpu().numpy().astype(np.float32) 
            b3_feats[idx:idx+self.B] = b3.detach().cpu().numpy().astype(np.float32)
            b4_feats[idx:idx+self.B] = b4.detach().cpu().numpy().astype(np.float32)
            b5_feats[idx:idx+self.B] = b5.detach().cpu().numpy().astype(np.float32) 
            fc1_feats[idx:idx+self.B] = fc1_out.detach().cpu().numpy().astype(np.float32) 
            fc2_feats[idx:idx+self.B] = fc2_out.detach().cpu().numpy().astype(np.float32)
            fc3_feats[idx:idx+self.B] = fc3_out.detach().cpu().numpy().astype(np.float32)
            file_order[idx:idx+self.B] = file_ids.detach().cpu().numpy().astype(np.float32)

            # obj_features = obj_features.detach().cpu().numpy().astype(np.float16)

            # feats.append(obj_features)
            # file_order.append(file_ids.detach().cpu().numpy())

            idx += 1*self.B

            plt.close('all')

            # if idx == 100:
            #     break
        
        # feats = np.concatenate(feats, axis=0)
        # file_order = np.concatenate(file_order, axis=0)
        # supcat_names = np.array(supcat_names)

        np.save(f'{output_dir}/vgg_b1_feats.npy', b1_feats)
        np.save(f'{output_dir}/vgg_b2_feats.npy', b2_feats)
        np.save(f'{output_dir}/vgg_b3_feats.npy', b3_feats)
        np.save(f'{output_dir}/vgg_b4_feats.npy', b4_feats)
        np.save(f'{output_dir}/vgg_b5_feats.npy', b5_feats)
        np.save(f'{output_dir}/vgg_fc1_feats.npy', fc1_feats)
        np.save(f'{output_dir}/vgg_fc2_feats.npy', fc2_feats)
        np.save(f'{output_dir}/vgg_fc3_feats.npy', fc3_feats)
        np.save(f'{output_dir}/vgg_file_order.npy', file_order)
        # np.save(f'{output_dir}/vgg_supcat.npy', np.array(supcat_names))
        # np.save(f'{output_dir}/vgg_cat.npy', np.array(cat_names))

        # tsne = TSNE(n_components=2).fit_transform(feats)
        # # pred_catnames_feats = [self.maskrcnn_to_ithor[i] for i in self.feature_obj_ids]

        # self.plot_by_classes(supcat_names, tsne, self.summ_writer)

        # # tsne plot colored by predicted labels
        # tsne_pred_figure = self.get_colored_tsne_image(supcat_names, tsne)
        # self.summ_writer.summ_figure(f'tsne/tsne_vgg_reduced_supcat', tsne_pred_figure)

        # tsne_pred_figure = self.get_colored_tsne_image(cat_names, tsne)
        # self.summ_writer.summ_figure(f'tsne/tsne_vgg_reduced_cat', tsne_pred_figure)

    def plot_tsne_only(self):

        self.summ_writer = utils.improc.Summ_writer(
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