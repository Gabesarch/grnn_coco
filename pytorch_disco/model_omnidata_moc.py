import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from backend import saverloader
from utils.improc import MetricLogger

# if hyp.do_lescroart_moc:
#     from backend import inputs_lescroart as inputs
# elif hyp.do_carla_moc:
#     from backend import inputs2 as inputs
# else:
#     assert(False)

# from backend import inputs2 as inputs
# from backend import inputs_lescroart as inputs2\

# %matplotlib inline
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import sys
sys.path.append('./omnidata/omnidata_tools/torch')
import os, sys; sys.path.append(os.path.expanduser(os.getcwd()))
from dataloader.pytorch_lightning_datamodule import OmnidataDataModule
from dataloader.pytorch3d_utils import *
from dataloader.viz_utils import *
os.environ['OMNIDATA_PATH'] = '/lab_data/tarrlab/gsarch/omnidata/omnidata_starter_dataset'
os.environ['OMNIDATA_CACHE_PATH'] = '/lab_data/tarrlab/scratch/gsarch/omnidata/omnidata_starter_dataset/.cache'

# from backend import inputs3 as inputs

from model_base import Model
# from nets.featnet2D import FeatNet2D
from nets.feat3dnet import Feat3dNet
from nets.emb3dnet import Emb3dNet
from nets.occnet import OccNet
# from nets.mocnet import MocNet
from nets.viewnet import ViewNet
# from nets.rgbnet import RgbNet

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

from tqdm import tqdm

# np.set_printoptions(precision=2)
# np.random.seed(0)

import ipdb
st = ipdb.set_trace

class OMNIDATA_MOC(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaMocModel()
        if hyp.do_feat3d and hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)

        if hyp.do_emb3d:
            # freeze the slow model
            self.model.feat3dnet_slow.eval()
            self.set_requires_grad(self.model.feat3dnet_slow, False)

        self.model.cuda()


    def get_dataloader(
        self, 

        ):

        dm = OmnidataDataModule(
            tasks = ['point_info', 'rgb', 'depth_euclidean', 'mask_valid'],
            train_datasets_to_options = dict(
                    # TaskonomyDataset = dict(data_amount='tiny', data_path='/datasets/omnidata/ta/'),
                    ReplicaDataset   = dict()
                ),
                eval_datasets_to_options  = dict(
                    # HypersimDataset   = dict(cooccurrence_method='FRAGMENTS')
                    ReplicaDataset   = dict()
                    # GSOReplicaDataset   = dict()
                    # TaskonomyDataset  = dict(data_amount='tiny')
                ),
                shared_options = dict(
                    data_amount  = 'debug',
                    data_path    = os.environ['OMNIDATA_PATH'],
                    cache_dir    = os.environ['OMNIDATA_CACHE_PATH'],
                    image_size   = 512,
                    multiview_sampling_method = 'FILENAME', # Works for Taskonomy, Replica
                    n_workers    = 4,
                    # force_refresh_tmp = True,
                    num_positive = hyp.S,
                ),
                train_options  = dict(),
                eval_options   = dict(),
                dataloader_kwargs = dict(batch_size=hyp.batch_sizes["train"]),
        )
        dm.setup(stage='val')
        dm.setup(stage='train')

        dl_val = dm.val_dataloader()[0]
        dl_train = dm.train_dataloader()

        dl = {}
        dl["train"] = dl_train
        dl["val"] = dl_val
        
        return dl
            
    # override go from base
    def go(self):
        self.start_time = time.time()
        self.initialize_model()
        print("------ Done creating models ------")

        if hyp.lr > 0:
            params_to_optimize = self.model.parameters()
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=hyp.lr)
        else:
            self.optimizer = None
        
        self.start_iter = saverloader.load_weights(self.model, self.optimizer)
        if hyp.start_at_iter1:
            self.start_iter = 1
        print("------ Done loading weights ------")
        # self.start_iter = 0

        all_inputs = self.get_dataloader()
        
        # set val to set 1
        hyp.valset = hyp.valset1
        if hyp.valset1:
            hyp.name = "%s_%s" % (hyp.name, hyp.valset1)
            hyp.sets_to_run['val'] = True
        else:
            hyp.sets_to_run['val'] = False
        hyp.valset_path = "%s/%s.txt" % (hyp.dataset_list_dir1, hyp.valset1)
        hyp.data_paths['val'] = hyp.valset_path
        hyp.data_formats['val'] = hyp.valset_format


        set_nums = []
        set_names = []
        set_seqlens = []
        set_batch_sizes = []
        set_inputs = []
        set_inputs2 = []
        set_inputs3 = []
        set_writers = []
        set_log_freqs = []
        set_do_backprops = []
        set_dicts = []
        set_loaders = []
        set_loaders2 = []
        set_loaders3 = []

        for set_name in hyp.set_names:
            if hyp.sets_to_run[set_name]:
                set_nums.append(hyp.set_nums[set_name])
                set_names.append(set_name)
                set_seqlens.append(hyp.seqlens[set_name])
                set_batch_sizes.append(hyp.batch_sizes[set_name])
                set_inputs.append(all_inputs[set_name])
                print(set_inputs[-1])
                #     set_inputs2.append(all_inputs2[set_name])
                #     set_inputs3.append(all_inputs3[set_name])
                # else:
                #     set_inputs2.append(None)
                #     set_inputs3.append(None)
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=1000000, flush_secs=1000000))
                set_log_freqs.append(hyp.log_freqs[set_name])
                set_do_backprops.append(hyp.sets_to_backprop[set_name])
                set_dicts.append({})
                set_loaders.append(iter(set_inputs[-1]))
                # if set_name in ["train"]:
                #     set_loaders2.append(iter(set_inputs2[-1]))
                #     set_loaders3.append(iter(set_inputs3[-1]))
                # else:
                #     set_loaders2.append(None)
                #     set_loaders3.append(None)

        
        metric_logger = MetricLogger(delimiter="  ")
        header = f'TRAIN | {hyp.name}'
        for step in metric_logger.log_every(range(self.start_iter+1, hyp.max_iters+1), 10, header):

            for (set_num,
                 set_name,
                 set_seqlen,
                 set_batch_size,
                 set_input,
                 set_writer,
                 set_log_freq,
                 set_do_backprop,
                 set_dict,
                 set_loader,
            ) in zip(
                set_nums,
                set_names,
                set_seqlens,
                set_batch_sizes,
                set_inputs,
                set_writers,
                set_log_freqs,
                set_do_backprops,
                set_dicts,
                set_loaders,
            ):   

                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0

                output_dict = dict()

                if log_this or set_do_backprop:
                    # print('%s: set_num %d; log_this %d; set_do_backprop %d; ' % (set_name, set_num, log_this, set_do_backprop))
                    # print('log_this = %s' % log_this)
                    # print('set_do_backprop = %s' % set_do_backprop)
                            
                    read_start_time = time.time()

                    # feed, _ = next(set_loader)

                    # print(set_name)

                    # dataloader_index = step % num_dataloaders
                    # if set_name in ["val", "test"]:
                    #     dataloader_index = 0


                    try:
                        feed = next(set_loader)
                    except StopIteration:
                        for i, (set_input) in enumerate(set_inputs):
                            #restart after one epoch. Note this does nothing for the tfrecord loader
                            set_loaders[i] = iter(set_input)
                        continue
                        # feed = next(set_loader)

                    visualize = False
                    if visualize:
                        

                        # # Show the RGB image
                        # show_batch_images(feed, batch_idx=0, view_idxs=[0,1,2], keys=['rgb'])

                        # show_batch_scene(feed, batch_idx=0, view_idxs=[0,1,2])
                        
                        batch_idx = 0
                        view_idxs=[0,1]
                        pos        = feed.get('positive', feed)
                        bpv        = pos['building'][batch_idx], pos['point'][batch_idx], pos['view'][batch_idx]
                        dataset    = pos['dataset'][batch_idx]
                        mask_valid = pos['mask_valid'].bool()[batch_idx,view_idxs]#.squeeze(1)
                        distance   = pos['depth_euclidean'][batch_idx,view_idxs]#.squeeze(1)
                        rgb        = pos['rgb'][batch_idx,view_idxs].unsqueeze(1)
                        cam_params = { k: v[batch_idx,view_idxs].unsqueeze(1)
                                    for (k, v) in get_batch_cam_params(pos).items()}
                        pcs_full   = batch_unproject_to_multiview_pointclouds(mask_valid=mask_valid, distance=distance, features=rgb, **cam_params)
                        cameras    = GenericPinholeCamera(
                                        R=cam_params['cam_to_world_R'].squeeze(1),
                                        T=cam_params['cam_to_world_T'].squeeze(1),
                                        K=cam_params['proj_K'].squeeze(1),
                                        K_inv=cam_params['proj_K_inv'].squeeze(1),
                                        device=distance.device)
                        pcs_full   = join_pointclouds_as_batch(pcs_full)

                        field = "points" 
                        field_list = [getattr(p, field + "_list")() for p in pcs_full]
                        field_list = torch.stack([f[0] for f in field_list])
                        # print(field_list)
                        # print(field_list.shape)
                        # print(len(field_list))
                        # print(field_list[0][0].shape)
                        # print(len(field_list[0]))

                        # print(pcs_full.shape)
                        # print(cameras.shape)

                    # extract output from omnidata
                    for k in feed['positive']:
                        # print(k)
                        # if isinstance(feed['positive'][k], dict):
                        #     feed['positive'][k] = {j:feed['positive'][k][j].cuda(non_blocking=True) for j in feed['positive'][k].keys()}
                        feed[k] = feed['positive'][k] #.cuda(non_blocking=True)
                    del feed['positive']

                    feed_cuda = {}
                    for k in feed:
                        try:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True)
                        except:
                            # some things are not tensors (e.g., filename)
                            feed_cuda[k] = feed[k]
                    read_time = time.time() - read_start_time

                    feed_cuda['writer'] = set_writer
                    feed_cuda['global_step'] = step
                    feed_cuda['set_num'] = set_num
                    feed_cuda['set_log_freq'] = set_log_freq
                    feed_cuda['set_name'] = set_name
                    feed_cuda['set_seqlen'] = set_seqlen
                    feed_cuda['set_batch_size'] = set_batch_size
                    
                    iter_start_time = time.time()
                    if set_do_backprop:
                        self.model.train()
                        loss, results, returned_early = self.model(feed_cuda)
                    else:
                        self.model.eval()
                        with torch.no_grad():
                            loss, results, returned_early = self.model(feed_cuda)
                    loss_py = loss.cpu().item()

                    if ((not returned_early) and 
                        (set_do_backprop) and 
                        (hyp.lr > 0)):
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    metric_logger.update(loss=loss)

                    if hyp.do_emb3d:
                        def update_slow_network(slow_net, fast_net, beta=0.999):
                            param_k = slow_net.state_dict()
                            param_q = fast_net.named_parameters()
                            for n, q in param_q:
                                if n in param_k:
                                    param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
                            slow_net.load_state_dict(param_k)
                        update_slow_network(self.model.feat3dnet_slow, self.model.feat3dnet)
                        
                    iter_time = time.time()-iter_start_time
                    total_time = time.time()-self.start_time

                    # print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (hyp.name,
                    #                                                                     step,
                    #                                                                     hyp.max_iters,
                    #                                                                     total_time,
                    #                                                                     read_time,
                    #                                                                     iter_time,
                    #                                                                     loss_py,
                    #                                                                     set_name))
                    if log_this:
                        set_writer.flush()
                    
            if hyp.do_save_outputs:
                out_fn = '%s_output_dict.npy' % (hyp.name)
                np.save(out_fn, output_dict)
                print('saved %s' % out_fn)
            
            if np.mod(step, hyp.snap_freq) == 0 and hyp.lr > 0:
                saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)

        for writer in set_writers: #close writers to flush cache into file
            writer.close()

            
class CarlaMocModel(nn.Module):
    def __init__(self):
        super(CarlaMocModel, self).__init__()

        # self.crop_guess = (18,18,18)
        # self.crop_guess = (2,2,2)
        self.crop = (18,18,18)
        
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_emb3d:
            self.emb3dnet = Emb3dNet()
            # make a slow net
            self.feat3dnet_slow = Feat3dNet(in_dim=4)
            # init slow params with fast params
            self.feat3dnet_slow.load_state_dict(self.feat3dnet.state_dict())
        # if hyp.do_rgb:
        #     self.rgbnet = RgbNet()
        if hyp.do_view:
            self.viewnet = ViewNet(feat_dim=hyp.feat3d_dim)

        if hyp.do_midas_depth_estimation:
            # midas depth estimation
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS") #, _use_new_zipfile_serialization=True)
            # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.midas.cuda()
            self.midas.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.default_transform
            
    def crop_feat(self, feat_pad, crop):
        Z_pad, Y_pad, X_pad = crop
        feat = feat_pad[:,:,
                        Z_pad:-Z_pad,
                        Y_pad:-Y_pad,
                        X_pad:-X_pad].clone()
        return feat
    
    def pad_feat(self, feat, crop):
        Z_pad, Y_pad, X_pad = crop
        feat_pad = F.pad(feat, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
        return feat_pad
    
    def prepare_common_tensors(self, feed, prep_summ=True):
        results = dict()
        
        if prep_summ:
            self.summ_writer = utils.improc.Summ_writer(
                writer=feed['writer'],
                global_step=feed['global_step'],
                log_freq=feed['set_log_freq'],
                fps=8,
                just_gif=feed['just_gif'],
            )
        else:
            self.summ_writer = None

        self.include_vis = True

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        
        self.rgb_camXs = feed["rgb"]-0.5
        self.depth_camXs = feed["depth_euclidean"]
        self.mask_valid_camXs = feed["mask_valid"]
        self.point_info = feed["point_info"]
        self.point = feed["point"]
        self.building = feed["building"]
        self.view = feed["view"]
        self.dataset = feed["dataset"]

        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z1, self.Y1, self.X1 = int(self.Z/1), int(self.Y/1), int(self.X/1)
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)
        self.W = self.rgb_camXs.shape[-2]
        self.H = self.rgb_camXs.shape[-1]

        view_idxs=list(np.arange(self.S))
        self.xyz_camRs = []
        self.pix_T_cams = []
        self.vox_utils = []
        self.occXs = []
        self.unpXs = []
        self.occ_halfmemX0_sup = []
        self.free_halfmemX0_sup = []
        for batch_idx in range(self.B):
            # pos        = feed.get('positive', feed)
            bpv        = feed['building'][batch_idx], feed['point'][batch_idx], feed['view'][batch_idx]
            dataset    = feed['dataset'][batch_idx]
            mask_valid = feed['mask_valid'].bool()[batch_idx,view_idxs]#.squeeze(1)
            distance   = feed['depth_euclidean'][batch_idx,view_idxs]#.squeeze(1)
            rgb        = feed['rgb'][batch_idx,view_idxs].unsqueeze(1)
            cam_params = { k: v[batch_idx,view_idxs].unsqueeze(1).cuda(non_blocking=True)
                        for (k, v) in get_batch_cam_params(feed).items()}
            # print("cam_params", cam_params)
            pcs_full   = batch_unproject_to_multiview_pointclouds(mask_valid=mask_valid, distance=distance, features=rgb, **cam_params)
            cameras    = GenericPinholeCamera(
                            R=cam_params['cam_to_world_R'].squeeze(1),
                            T=cam_params['cam_to_world_T'].squeeze(1),
                            K=cam_params['proj_K'].squeeze(1),
                            K_inv=cam_params['proj_K_inv'].squeeze(1),
                            device=distance.device)
            pcs_full   = join_pointclouds_as_batch(pcs_full)

            print(distance.shape)
            xyz_camX = cameras.unproject_metric_depth_euclidean(distance.squeeze(1), world_coordinates=False).reshape(len(cameras), -1, 3)
            # xyz_camX = [pts[keep.reshape(-1)]   for pts, keep in zip(xyz_camX, mask_valid)]

            world_to_view_transform = cameras.get_world_to_view_transform().get_matrix()
            view_to_world_transform = cameras.get_world_to_view_transform().inverse().get_matrix()
            # print("world_to_view_transform", world_to_view_transform)
            # print("view_to_world_transform", view_to_world_transform)
            # camX_T_camW = utils.geom.merge_rt(cam_params['cam_to_world_R'].squeeze(1), cam_params['cam_to_world_T'].squeeze(1))
            # camW_T_camX = utils.geom.safe_inverse(camX_T_camW)
            # print("camX_T_camW", camX_T_camW)
            # print("camW_T_camX", camW_T_camX)
            # assert(False)

            field = "points" 
            field_list = [getattr(p, field + "_list")() for p in pcs_full]
            # field_list = torch.stack([f[0] for f in field_list])
            field_list = [f[0] for f in field_list]
            print(field_list[0].shape)
            print(field_list[0])

            xyz_maxs = torch.max(field_list[0], dim=0)[0]
            xyz_mins = torch.min(field_list[0], dim=0)[0]
            xyz_means = torch.mean(field_list[0], dim=0).cpu().numpy()
            # print(xyz_means)
            # print(xyz_maxs)
            # print(xyz_mins)

            pix_T_cams = cam_params['proj_K'].squeeze(1)
            self.pix_T_cams.append(pix_T_cams)
            
            # camX_T_camW = utils.geom.merge_rt(cam_params['cam_to_world_R'].squeeze(1), cam_params['cam_to_world_T'].squeeze(1))
            # camW_T_camX = utils.geom.safe_inverse(camX_T_camW)
            
            pixX0_T_camW = utils.basic.matmul2(pix_T_cams, view_to_world_transform)

            scene_centroid_x_noise = np.random.normal(0, 0.2) #+ offset_x
            scene_centroid_y_noise = np.random.normal(0, 0.2) #+ offset_y
            scene_centroid_z_noise = np.random.normal(0, 0.2) #+ offset_z

            xyz_max = int(torch.clip(max(xyz_maxs), max=20.0).cpu().numpy()) / 2.
            scene_centroid_B = torch.tensor([int(scene_centroid_x_noise+xyz_means[0]), int(scene_centroid_y_noise+xyz_means[1]), int(xyz_means[2]+scene_centroid_z_noise)]).cuda().squeeze(0).reshape(1,3)
            bounds_B = torch.tensor([-xyz_max, xyz_max, -xyz_max, xyz_max, -xyz_max, xyz_max]).cuda()

            # self.vox_utils.append(utils.vox.Vox_util(self.Z, self.Y, self.X, feed['set_name'], scene_centroid_B, bounds=bounds_B, assert_cube=True))
            self.vox_utils.append(utils.vox.Vox_util(self.Z, self.Y, self.X, feed['set_name'], scene_centroid_B, assert_cube=True))
            occXs_ = []
            unpXs_ = []
            occ_halfmemX0_sup = []
            free_halfmemX0_sup = []
            
            for s in range(self.S):
                occXs = self.vox_utils[-1].voxelize_xyz(field_list[s].unsqueeze(0), self.Z, self.Y, self.X)
                occXs_.append(occXs)
                
                unpXs = self.vox_utils[-1].unproject_rgb_to_mem(
                    self.rgb_camXs[batch_idx, s].unsqueeze(0), self.Z, self.Y, self.X, pixX0_T_camW[s].unsqueeze(0))
                unpXs_.append(unpXs)

            occ_halfmemX0_sup_, free_halfmemX0_sup_, _, _ = self.vox_utils[-1].prep_occs_supervision(
                view_to_world_transform.unsqueeze(0),
                xyz_camX.unsqueeze(0),
                self.Z4, self.Y4, self.X4,
                agg=True)

            self.occXs.append(torch.stack(occXs_))
            self.unpXs.append(torch.stack(unpXs_))

            self.occ_halfmemX0_sup.append(occ_halfmemX0_sup_)
            self.free_halfmemX0_sup.append(free_halfmemX0_sup_)   

            if batch_idx==0 and self.summ_writer.save_this:

                def _broadcast_bmm(a, b) -> torch.Tensor:
                    """
                    Batch multiply two matrices and broadcast if necessary.

                    Args:
                        a: torch tensor of shape (P, K) or (M, P, K)
                        b: torch tensor of shape (N, K, K)

                    Returns:
                        a and b broadcast multiplied. The output batch dimension is max(N, M).

                    To broadcast transforms across a batch dimension if M != N then
                    expect that either M = 1 or N = 1. The tensor with batch dimension 1 is
                    expanded to have shape N or M.
                    """
                    if a.dim() == 2:
                        a = a[None]
                    if len(a) != len(b):
                        if not ((len(a) == 1) or (len(b) == 1)):
                            msg = "Expected batch dim for bmm to be equal or 1; got %r, %r"
                            raise ValueError(msg % (a.shape, b.shape))
                        if len(a) == 1:
                            a = a.expand(len(b), -1, -1)
                        if len(b) == 1:
                            b = b.expand(len(a), -1, -1)
                    return a.bmm(b)

                def transform_points(points, composed_matrix, eps: Optional[float] = None) -> torch.Tensor:
                    """
                    Use this transform to transform a set of 3D points. Assumes row major
                    ordering of the input points.

                    Args:
                        points: Tensor of shape (P, 3) or (N, P, 3)
                        eps: If eps!=None, the argument is used to clamp the
                            last coordinate before performing the final division.
                            The clamping corresponds to:
                            last_coord := (last_coord.sign() + (last_coord==0)) *
                            torch.clamp(last_coord.abs(), eps),
                            i.e. the last coordinates that are exactly 0 will
                            be clamped to +eps.

                    Returns:
                        points_out: points of shape (N, P, 3) or (P, 3) depending
                        on the dimensions of the transform
                    """
                    points_batch = points.clone()
                    if points_batch.dim() == 2:
                        points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
                    if points_batch.dim() != 3:
                        msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
                        raise ValueError(msg % repr(points.shape))

                    N, P, _3 = points_batch.shape
                    ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
                    points_batch = torch.cat([points_batch, ones], dim=2)

                    # composed_matrix = self.get_matrix()
                    points_out = _broadcast_bmm(points_batch, composed_matrix)
                    denom = points_out[..., 3:]  # denominator
                    if eps is not None:
                        denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
                        denom = denom_sign * torch.clamp(denom.abs(), eps)
                    points_out = points_out[..., :3] / denom

                    # When transform is (1, 4, 4) and points is (P, 3) return
                    # points_out of shape (P, 3)
                    if points_out.shape[0] == 1 and points.dim() == 2:
                        points_out = points_out.reshape(points.shape)

                    return points_out

                # plot occupancy supervision
                # camRs_T_camXs_ = __p(cameras.get_world_to_view_transform().inverse().get_matrix().unsqueeze(0))
                camRs_T_camXs = cameras.get_world_to_view_transform().inverse().get_matrix()
                # xyz_camXs_ = __p(xyz_camX.unsqueeze(0))
                xyz_camRs = cameras.get_world_to_view_transform().inverse().transform_points(xyz_camX)
                print(xyz_camRs.shape)
                print(xyz_camRs[0])
                xyz_camRs = transform_points(xyz_camX, cameras.get_world_to_view_transform().inverse().get_matrix())
                print(xyz_camRs.shape)
                print(xyz_camRs[0])
                # assert(False)
                # print(camRs_T_camXs_)
                # print(camRs_T_camXs.shape)
                # print(xyz_camX.shape)
                xyz_camRs = utils.geom.apply_4x4(camRs_T_camXs, xyz_camX)
                print(mask_valid.shape)
                xyz_camRs = [pts[keep.reshape(-1)]   for pts, keep in zip(xyz_camRs, mask_valid)]
                # print(xyz_camXs_.shape, camRs_T_camXs_.shape)
                # B, N, _ = list(xyz_camXs_.shape)
                # ones = torch.ones_like(xyz_camXs_[:,:,0:1])
                # xyz1 = torch.cat([xyz_camXs_, ones], 2)
                # xyz1_t = torch.transpose(xyz_camXs_, 1, 2)
                # # this is B x 4 x N
                # xyz2_t = torch.matmul(camRs_T_camXs_, xyz1_t)
                # xyz2 = torch.transpose(xyz2_t, 1, 2)
                # xyz2 = xyz2[:,:,:3]
                print(xyz_camRs[0].shape)
                print(xyz_camRs[0])
                assert(False)
                occXs_ = self.vox_utils[-1].voxelize_xyz(xyz_camXs_, self.Z4, self.Y4, self.X4)
                occRs_ = self.vox_utils[-1].voxelize_xyz(xyz_camRs_, self.Z4, self.Y4, self.X4)
                freeXs_ = self.vox_utils[-1].get_freespace(xyz_camXs_, occXs_)
                freeRs_ = self.vox_utils[-1].apply_4x4_to_vox(camRs_T_camXs_, freeXs_)
                occXs = __u(occXs_)
                occRs = __u(occRs_)
                freeXs = __u(freeXs_)
                freeRs = __u(freeRs_)

                # print(occXs.shape, occRs.shape, freeXs.shape, freeRs.shape)
                # print(self.occXs[-1].squeeze().unsqueeze(0).shape)

                self.summ_writer.summ_occs('occ/occX0s_input', self.occXs[-1].squeeze().unsqueeze(0).unsqueeze(2).unbind(1), reduce_axes=[3])
                # self.summ_writer.summ_occs('occ/occXs_input', occXs.unbind(1), reduce_axes=[3])
                # self.summ_writer.summ_occs('occ/occRs_input', occRs.unbind(1), reduce_axes=[3])
                self.summ_writer.summ_occ('occ/occR0_input', occRs[:,0], reduce_axes=[3])
                self.summ_writer.summ_occ('occ/occR1_input', occRs[:,1], reduce_axes=[3])
                self.summ_writer.summ_occ('occ/occR_input_combined', torch.cat(occRs.unbind(1), dim=3), reduce_axes=[3])
                # self.summ_writer.summ_occs('occ/freeXs_input', freeXs.unbind(1), reduce_axes=[3])
                # self.summ_writer.summ_occs('occ/freeRs_input', freeRs.unbind(1), reduce_axes=[3])

            # self.xyz_camRs.append(field_list)
            # print(field_list.shape)
        self.occXs = torch.stack(self.occXs).squeeze(2)
        self.unpXs = torch.stack(self.unpXs).squeeze(2)
        self.occ_halfmemX0_sup = torch.cat(self.occ_halfmemX0_sup, dim=0)
        self.free_halfmemX0_sup = torch.cat(self.free_halfmemX0_sup, dim=0)

        all_ok = True
        return all_ok

    # def apply_vox_util_to_batch(self, function):
    #     if function=="get_inbounds":

    def run_train(self, feed):
        use_camR = False
        results = dict()

        global_step = feed['global_step']
        set_name = feed['set_name']
        # print("SET_NAME:", set_name)
        total_loss = torch.tensor(0.0).cuda()
        # st()

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        # since we are using multiview data, all Rs are aligned
        # we'll encode X0 with the fast net, then warp to R
        # we'll use the X0 version for occ loss, to get max labels
        # we'll encode X1/R1 with slow net, and use this for emb loss

        self.summ_writer.summ_rgbs(f'inputs/rgbs', self.rgb_camXs.unbind(1))
        self.summ_writer.summ_unps('inputs/unps', self.unpXs.unbind(1), self.occXs.unbind(1))
        
        if hyp.do_feat3d:
            assert(self.S==2)

            featX0_input = torch.cat([self.occXs[:,1:2], self.occXs[:,1:2]*self.unpXs[:,1:2]], dim=2)
            featX0_input_ = __p(featX0_input)
            feat3d_loss, feat_halfmemX0, _ = self.feat3dnet(
                featX0_input_,
                summ_writer=self.summ_writer,
                # set_name=set_name,
            )
            total_loss += feat3d_loss

            valid_halfmemX0 = torch.ones_like(feat_halfmemX0[:,0:1])

            # self.summ_writer.summ_feat(f'feat3d/feat_halfmemX0', feat_halfmemX0[0], valid=valid_halfmemX0[0], pca=True)

            if hyp.do_emb3d:

                feat_memX0_input = torch.cat([self.occXs[:,0:1], self.occXs[:,0:1]*self.unpXs[:,0:1]], dim=2)
                feat_memX0_input = __p(feat_memX0_input)
                _, altfeat_halfmemX0, _ = self.feat3dnet_slow(feat_memX0_input)
                altvalid_halfmemX0 = torch.ones_like(altfeat_halfmemX0[:,0:1])
                self.summ_writer.summ_feat(f'feat3d/altfeat_input', feat_memX0_input, pca=True)
                self.summ_writer.summ_feat(f'feat3d/altfeat_output', altfeat_halfmemX0, valid=altvalid_halfmemX0, pca=True)
                # self.summ_writer.summ_oned(f'feat3d/altvalid_halfmemX0', altvalid_halfmemX0, bev=True, norm=False)

        if hyp.do_occ:

            self.occ_halfmemX0_sup = __p(self.occ_halfmemX0_sup)
            self.free_halfmemX0_sup = __p(self.free_halfmemX0_sup)
            
            # be more conservative with "free"
            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            self.free_halfmemX0_sup = 1.0 - (F.conv3d(1.0 - self.free_halfmemX0_sup, weights, padding=1)).clamp(0, 1)
            
            occ_loss, occ_memX0_pred = self.occnet(
                feat_halfmemX0, 
                occ_g=self.occ_halfmemX0_sup, 
                free_g=self.free_halfmemX0_sup, 
                valid=valid_halfmemX0.squeeze(1),
                summ_writer=self.summ_writer,
                # set_name=set_name,
                )
            total_loss += occ_loss

            # self.summ_writer.summ_occ(name, occ, reduce_axes=[3], only_return=False)

            # print(self.occ_halfmemX0_sup.shape)
            # occ_vis = self.summ_writer.summ_occ(f'occ/occ_g', self.occ_halfmemX0_sup.unsqueeze(1), only_return=True)
            # plt.figure(0); plt.clf()
            # plt.imshow(occ_vis.squeeze(0).permute(1,2,0))
            # plt.savefig('images/test.png')
            # print("occ_vis", occ_vis.shape)
            # plt.figure(0); plt.clf()
            # print(self.rgb_camXs.shape)
            # plt.imshow(self.rgb_camXs[0,0].permute(1,2,0).cpu().numpy())
            # plt.savefig('images/test2.png')
            # plt.figure(0); plt.clf()
            # print(self.rgb_camXs.shape)
            # plt.imshow(self.rgb_camXs[0,1].permute(1,2,0).cpu().numpy())
            # plt.savefig('images/test3.png')
            # time.sleep(15)
            
        if hyp.do_emb3d:
            emb3d_loss = self.emb3dnet(
                feat_halfmemX0,
                altfeat_halfmemX0,
                valid_halfmemX0.round(),
                altvalid_halfmemX0.round(),
                self.summ_writer)
            total_loss += emb3d_loss

        if hyp.do_view:
            # featX1 = self.vox_util.apply_4x4_to_vox(self.camXs_T_camX0s[:, 1], feat_halfmemX0[:,1:2]) # get features from camX1
            assert(hyp.do_feat3d)
            # we warped the features into the canonical view
            # now we resample to the target view and decode
            PH, PW = self.H//2, self.W//2
            sy = float(PH)/float(self.H)
            sx = float(PW)/float(self.W)
            assert(sx==0.5) # else we need a fancier downsampler
            assert(sy==0.5)
            projpix_T_cams = __u(utils.geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))
            assert(self.S==2) # else we should warp each feat in 1: 
            feat_projX00 = self.vox_util.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camX0s_T_camXs[:,1], feat_halfmemX1, # use feat1 to predict rgb0
                hyp.view_depth, PH, PW)
            rgb_X00 = utils.basic.downsample(self.rgb_camXs[:,0], 2)
            valid_X00 = utils.basic.downsample(self.valid_camXs[:,0], 2)
            # decode the perspective volume into an image
            # print(feat_halfmemX1.shape)
            view_loss, rgb_e, emb2D_e = self.viewnet(
                feat_projX00,
                rgb_X00,
                valid_X00,
                self.summ_writer,
                f'rgb',
                # set_name=set_name,
                )      
            # print(rgb_e.shape)              
            total_loss += view_loss

        self.summ_writer.summ_scalar(f'loss', total_loss.cpu().item())

        return total_loss, results, False

    def forward(self, feed):
        
        set_name = feed['set_name']
        if set_name=='test':
            just_gif = True
        else:
            just_gif = False
        feed['just_gif'] = just_gif
        
        ok = self.prepare_common_tensors(feed)
        if ok:
            if set_name=='train' or set_name=='val':
                return self.run_train(feed)
            else:
                return False # no test here yet
        else:
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, False
        
        # # arriving at this line is bad
        # print('weird set_name:', set_name)
        # assert(False)