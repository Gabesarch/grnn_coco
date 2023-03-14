import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from backend import saverloader
from utils.improc import MetricLogger

# %matplotlib inline
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import sys
sys.path.append('./omnidata/omnidata_tools/torch')
import os, sys; sys.path.append(os.path.expanduser(os.getcwd()))
from dataloader.pytorch_lightning_datamodule import OmnidataDataModule
from dataloader.pytorch3d_utils import *
from dataloader.viz_utils import *
os.environ['OMNIDATA_PATH'] = '/lab_data/tarrlab/scratch/gsarch/omnidata/omnidata_starter_dataset2'
os.environ['OMNIDATA_CACHE_PATH'] = '/lab_data/tarrlab/scratch/gsarch/omnidata/omnidata_starter_dataset/.cache'

from model_base import Model

use_enc3d = True
if use_enc3d:
    from nets.feat3dnet import Feat3dNet
else:
    from nets.featnet import FeatNet as Feat3dNet

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

from   pytorch3d.transforms import Rotate, Transform3d, Translate
from   pytorch3d.transforms import euler_angles_to_matrix

from tqdm import tqdm

# fix the seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class OMNIDATA_MOC(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaMocModel()
        if hyp.do_feat3d and hyp.do_freeze_feat3d:
            assert(False) # why are we freezing?
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)

        if hyp.do_emb3d:
            print('freezing feat3dnet_slow..')
            # freeze the slow model
            self.model.feat3dnet_slow.eval()
            self.set_requires_grad(self.model.feat3dnet_slow, False)

        self.model.cuda()


    def get_dataloader(
        self, 
        ):

        print("BATCH SIZE:", hyp.batch_sizes["train"])
        if hyp.debug:
            data_amount = 'debug'
            max_samples = hyp.max_samples
            datasets_train = dict(
                    # HypersimDataset   = dict(),
                    TaskonomyDataset = dict(),
                    # ReplicaDataset   = dict(),
                    # GSOReplicaDataset   = dict(),
                )
            datasets_val = dict(
                    # HypersimDataset   = dict(),
                    TaskonomyDataset = dict(),
                    # ReplicaDataset   = dict(),
                    # GSOReplicaDataset   = dict(),
                )
        else:
            data_amount = 'tiny'
            max_samples = hyp.max_samples
            datasets_train = dict(
                    HypersimDataset   = dict(),
                    TaskonomyDataset = dict(),
                    ReplicaDataset   = dict(),
                    GSOReplicaDataset   = dict(),
                )
            datasets_val = dict(
                    HypersimDataset   = dict(),
                    TaskonomyDataset = dict(),
                    ReplicaDataset   = dict(),
                    GSOReplicaDataset   = dict(),
                )
        dm = OmnidataDataModule(
                tasks = ['point_info', 'rgb', 'depth_euclidean', 'mask_valid'],
                train_datasets_to_options = datasets_train,
                eval_datasets_to_options  = datasets_val,
                shared_options = dict(
                    # data_amount  = 'debug',
                    # data_amount  = 'tiny',
                    # data_amount  = 'fullplus',
                    max_samples = max_samples,
                    data_amount = data_amount,
                    data_path    = os.environ['OMNIDATA_PATH'],
                    cache_dir    = os.environ['OMNIDATA_CACHE_PATH'],
                    image_size   = 512,
                    multiview_sampling_method = 'CENTER_VISIBLE', #'FILENAME', # Works for Taskonomy, Replica
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

        print(f"Length train: {len(dl_train)}")
        print(f"Length val: {len(dl_val)}")

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
        
        # # set val to set 1
        # hyp.valset = hyp.valset1
        # if hyp.valset1:
        #     hyp.name = "%s_%s" % (hyp.name, hyp.valset1)
        #     hyp.sets_to_run['val'] = True
        # else:
        #     hyp.sets_to_run['val'] = False
        # hyp.valset_path = "%s/%s.txt" % (hyp.dataset_list_dir1, hyp.valset1)
        # hyp.data_paths['val'] = hyp.valset_path
        # hyp.data_formats['val'] = hyp.valset_format


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

        print(hyp.set_names)

        for set_name in hyp.set_names:
            if hyp.sets_to_run[set_name]:
                set_nums.append(hyp.set_nums[set_name])
                set_names.append(set_name)
                set_seqlens.append(hyp.seqlens[set_name])
                set_batch_sizes.append(hyp.batch_sizes[set_name])
                set_inputs.append(all_inputs[set_name])
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=1000000, flush_secs=1000000))
                set_log_freqs.append(hyp.log_freqs[set_name])
                set_do_backprops.append(hyp.sets_to_backprop[set_name])
                set_dicts.append({})
                set_loaders.append(iter(set_inputs[-1]))

        
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
                        print("STOPITERATION!")
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

                    # extract output from omnidata
                    for k in feed['positive']:
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
                    # loss_py = loss.cpu().item()

                    if ((not returned_early) and 
                        (set_do_backprop) and 
                        (hyp.lr > 0) and
                        (loss is not None)):
                        
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

        if len(self.rgb_camXs)<self.B:
            all_ok = False
            return all_ok

        self.depth_camXs = feed["depth_euclidean"].squeeze(2)
        self.mask_valid_camXs = feed["mask_valid"].squeeze(2).bool()
        self.point_info = feed["point_info"]
        self.point = feed["point"]
        self.building = feed["building"]
        self.view = feed["view"]
        self.dataset = feed["dataset"]
        self.W = self.rgb_camXs.shape[-2]
        self.H = self.rgb_camXs.shape[-1]

        # print(self.dataset)

        cam_params = { k: v.cuda(non_blocking=True)
                        for (k, v) in get_batch_cam_params(feed).items()}
        # self.fov = cam_params['fov']

        # pix_T_camX = np.array([
        #     [(self.W/2.)*1 , 0., 0., 0.],
        #     [0., (self.H/2.)*1, 0., 0.],
        #     [0., 0.,  1, 0],
        #     [0., 0., 0, 1]])
        # pix_T_camX[0,2] = self.W/2.
        # pix_T_camX[1,2] = self.H/2.
        # self.pix_T_cams = torch.from_numpy(pix_T_camX).cuda().float().unsqueeze(0).unsqueeze(0).expand(self.B,self.S,4,4).clone()

        # tan_hfov = torch.tan(self.fov / 2.)
        # # pix_T_cams_ = self.pix_T_cams.clone()
        # for b in range(self.B):
        #     for s in range(self.S):
        #         self.pix_T_cams[b,s,0,0] /= tan_hfov[b,s]
        #         self.pix_T_cams[b,s,1,1] /= tan_hfov[b,s]

        mask_valid = __p(feed['mask_valid'].bool()) #[batch_idx,view_idxs]#.squeeze(1)
        depth   = __p(feed['depth_euclidean']) #[batch_idx,view_idxs] #.squeeze(1)
        rgb        = feed['rgb'] #[batch_idx,view_idxs].unsqueeze(1)
        cam_params = { k: __p(v).unsqueeze(1).cuda(non_blocking=True)
                    for (k, v) in get_batch_cam_params(feed).items()}
        # print("cam_params", cam_params)
        pcs_full   = batch_unproject_to_multiview_pointclouds(mask_valid=mask_valid, distance=depth, features=rgb, **cam_params)
        field = "points" 
        field_list = [getattr(p, field + "_list")() for p in pcs_full]
        field_list = [f[0] for f in field_list]

        if len(field_list[0])==0:
            all_ok = False
            return all_ok

        cameras    = GenericPinholeCamera(
                        R=cam_params['cam_to_world_R'].squeeze(1),
                        T=cam_params['cam_to_world_T'].squeeze(1),
                        K=cam_params['proj_K'].squeeze(1),
                        K_inv=cam_params['proj_K_inv'].squeeze(1),
                        device=depth.device)
        xyz_camX = cameras.unproject_metric_depth_euclidean(depth.squeeze(1), world_coordinates=False).reshape(len(cameras), -1, 3)
        # print(xyz_camX.shape)

        # normalize PC
        # xyz_camX_u = __u(xyz_camX)
        # for b in range(self.B):
        #     xyz_camX_u[b] = xyz_camX_u[b] - torch.median(xyz_camX_u[b,0],dim=0).values.unsqueeze(0).unsqueeze(0)
        # xyz_camX = __p(xyz_camX_u)

        # subtract_median = True
        # if subtract_median:
        #     xyz_camX = xyz_camX - torch.median

        world_to_view_transform = cameras.get_world_to_view_transform()
        # print(xyz_camX.shape)
        # xyz_camX = xyz_camX[:,:,[0,2,1]]
        xyz_origin = world_to_view_transform.inverse().transform_points(xyz_camX) #.reshape((batch_size, height, width, 3))

        self.xyz_camXs = __u(xyz_camX)
        self.origin_T_camXs = __u(world_to_view_transform.inverse().get_matrix())
        self.xyz_camRs = __u(xyz_origin)
        self.depth_camXs = __u(depth)
        self.mask_valid = __u(mask_valid)

        # print(self.depth_camXs.shape)
        # depth_vis = self.depth_camXs.cpu().numpy()[0,0].squeeze()
        # depth_vis[depth_vis>20.] = 0
        # print(1, np.max(depth_vis))
        # plt.figure(1); plt.clf()
        # plt.imshow(depth_vis)
        # plt.colorbar()
        # plt.savefig('images/test.png')
        # assert(False)


        # HypersimDataset axes are different than the rest - HACK to adjust for this
        for batch_idx in range(self.B):
            if self.dataset[batch_idx]=="HypersimDataset":
                self.xyz_camXs[batch_idx] = self.xyz_camXs[batch_idx,:,:,[0,2,1]]
                self.xyz_camRs[batch_idx] = self.xyz_camRs[batch_idx,:,:,[0,2,1]]
                # print(xyz_camX.shape)
                # print(xyz_origin.shape)

        degrees_rotate = torch.rand(self.B, dtype=torch.double, device='cuda') * 179  # degrees to rotate the second view
        euler_angles = torch.zeros((self.B, 3), dtype=torch.double, device='cuda') #torch.tensor([(np.zeros(self.B), np.zeros(self.B), np.radians(degrees_rotate))], dtype=torch.double, device='cuda')
        euler_angles[:,2] = degrees_rotate
        R_pc_transform = euler_angles_to_matrix(euler_angles, 'XZY')

        T_pc_transform = torch.tensor([(0., 0., 0.)], dtype=torch.double, device='cuda').repeat(self.B,1) #torch.rand((self.B, 3), dtype=torch.double, device='cuda') 

        RT_pc_transform = Rotate(R_pc_transform, device=R_pc_transform.device).compose(Translate(T_pc_transform, device=R_pc_transform.device))

        euler_angles = torch.zeros((self.B, 3), dtype=torch.double, device='cuda') #torch.tensor([(np.zeros(self.B), np.zeros(self.B), np.radians(degrees_rotate))], dtype=torch.double, device='cuda')
        euler_angles[:,1] = degrees_rotate
        R_ = utils.geom.eul2rotm(*euler_angles.unbind(1))
        camX1_T_camX0 = utils.geom.merge_rt(R_, T_pc_transform).unsqueeze(1)
        camX0_T_camX0 = utils.geom.eye_4x4s(self.B, 1)
        self.camX0s_T_camXs = torch.cat([camX0_T_camX0, camX1_T_camX0], dim=1)
        self.camXs_T_camX0s = __u(utils.geom.safe_inverse(__p(self.camX0s_T_camXs)))
        
        self.xyz_camXs = torch.stack([self.xyz_camRs[:,0], RT_pc_transform.transform_points(self.xyz_camRs[:,1])], dim=1)

        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z1, self.Y1, self.X1 = int(self.Z/1), int(self.Y/1), int(self.X/1)
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)
        
        view_idxs=list(np.arange(self.S))
        self.vox_utils = []
        self.occXs = []
        self.unpXs = []
        self.occ_halfmemX0_sup = []
        self.free_halfmemX0_sup = []
        for batch_idx in range(self.B):
            # aggregated_xyz_camX0s = torch.cat(self.xyz_camRs[batch_idx].unbind(0), dim=0)
            # print(self.mask_valid)
            aggregated_xyz_camX0s = []
            for s in range(self.S):
                aggregated_xyz_camX0s.append(self.xyz_camXs[batch_idx,s,self.mask_valid[batch_idx,s].flatten(),:])
            # aggregated_xyz_camX0s = torch.cat(self.xyz_camXs[batch_idx].unbind(0), dim=0)
            aggregated_xyz_camX0s = torch.cat(aggregated_xyz_camX0s, dim=0)

            if len(aggregated_xyz_camX0s)==0:
                all_ok = False
                return all_ok
            # where_valid = torch.where(aggregated_xyz_camX0s<70.)
            # print(where_valid)
            # print(aggregated_xyz_camX0s.shape)
            # aggregated_xyz_camX0s = aggregated_xyz_camX0s[where_valid[0], where_valid[1]]
            xyz_means = torch.median(aggregated_xyz_camX0s, dim=0).values.cpu().numpy()
            # xyz_means = torch.mean(aggregated_xyz_camX0s, dim=0).cpu().numpy()

            scene_centroid_x_noise = np.random.normal(0, 0.5) #+ offset_x
            scene_centroid_y_noise = np.random.normal(0, 0.5) #+ offset_y
            scene_centroid_z_noise = np.random.normal(0, 0.5) #+ offset_z
            
            scene_centroid_B = torch.tensor([scene_centroid_x_noise+xyz_means[0], scene_centroid_y_noise+xyz_means[1], xyz_means[2]+scene_centroid_z_noise]).cuda().squeeze(0).reshape(1,3)
            assert(not (hyp.use_bounds and hyp.use_tight_bounds))
            if hyp.use_tight_bounds:
                # mask = torch.where(aggregated_xyz_camX0s<20.)
                bounds_aggregated_xyz_camX0s = torch.abs(aggregated_xyz_camX0s)
                bounds_aggregated_xyz_camX0s[bounds_aggregated_xyz_camX0s>20.] = -9999
                xyz_argmax = torch.argmax(bounds_aggregated_xyz_camX0s, dim=0)
                xyz_maxs = torch.tensor([abs(aggregated_xyz_camX0s[xyz_argmax[0],0]), abs(aggregated_xyz_camX0s[xyz_argmax[1],1]), abs(aggregated_xyz_camX0s[xyz_argmax[2],2])])
                xyz_max = torch.clip(xyz_maxs, max=20.0).cpu().numpy() / 2.
                bounds_B = torch.tensor([-float(xyz_max[0]), float(xyz_max[0]), -float(xyz_max[1]), float(xyz_max[1]), -float(xyz_max[2]), float(xyz_max[2])]).cuda()
                # print(bounds_B)
                self.vox_utils.append(utils.vox.Vox_util(self.Z, self.Y, self.X, feed['set_name'], scene_centroid_B, bounds=bounds_B, assert_cube=False))
            elif hyp.use_bounds:
                if False:
                    depth_vis = self.depth_camXs[batch_idx].reshape(-1)
                    depth_vis[depth_vis>20.] = 0
                    xyz_max = torch.max(depth_vis)
                else:
                    # print(xyz_means)
                    # print(scene_centroid_B)
                    # print(aggregated_xyz_camX0s[:,2])
                    bounds_aggregated_xyz_camX0s = torch.abs(aggregated_xyz_camX0s - scene_centroid_B)
                    # print(bounds_aggregated_xyz_camX0s[:,0])
                    where_mask = torch.where(bounds_aggregated_xyz_camX0s>20.)
                    # print(len(where_mask))
                    bounds_aggregated_xyz_camX0s[where_mask[0], where_mask[1]] = -9999
                    # print(bounds_aggregated_xyz_camX0s.shape)
                    xyz_max = torch.max(bounds_aggregated_xyz_camX0s.flatten())
                    # print(xyz_argmax)
                    # xyz_max = abs(aggregated_xyz_camX0s.flatten()[xyz_argmax])
                    # print(torch.max(bounds_aggregated_xyz_camX0s, dim=0))
                    # print(xyz_max)
                xyz_max = float(torch.clip(xyz_max, max=20.0).cpu().numpy())
                bounds_B = torch.tensor([-xyz_max, xyz_max, -xyz_max, xyz_max, -xyz_max, xyz_max]).cuda()
                # print(bounds_B)
                self.vox_utils.append(utils.vox.Vox_util(self.Z, self.Y, self.X, feed['set_name'], scene_centroid_B, bounds=bounds_B, assert_cube=hyp.assert_cube))
            else:
                self.vox_utils.append(utils.vox.Vox_util(self.Z, self.Y, self.X, feed['set_name'], scene_centroid_B, assert_cube=hyp.assert_cube))
            
            if True:
                occXs, vox_inds = self.vox_utils[-1].voxelize_xyz(self.xyz_camXs[batch_idx], self.Z, self.Y, self.X, return_vox_ind=True)
                rgb_masked = self.rgb_camXs[batch_idx].reshape(self.S,3,-1).permute(1,0,2).reshape(3,-1)
                unpXs = torch.zeros([self.S, 3, self.Z, self.Y, self.X], device=torch.device('cuda')).float()
                # vox_inds = vox_inds.reshape(self.S, -1)
                unpXs = unpXs.permute(1,0,2,3,4).reshape(3,self.S*self.Z*self.Y*self.X)
                unpXs[:, vox_inds.long()] = rgb_masked
                # out of bounds gets mapped to 0,0,0
                N = self.xyz_camXs[batch_idx].shape[1]
                base = torch.arange(0, self.S, dtype=torch.int32, device=torch.device('cuda'))*self.X * self.Y * self.Z
                base = torch.reshape(base, [self.S, 1]).repeat([1, N]).view(self.S*N)
                # zero out the singularity
                unpXs[:, base.long()] = 0.0

                unpXs = unpXs.reshape(3,self.S,self.Z,self.Y,self.X).permute(1,0,2,3,4)
                unpXs = unpXs.reshape(self.S, 3,self.Z,self.Y,self.X).unsqueeze(0)
                occXs = occXs.unsqueeze(0)
            else:
                occXs = self.vox_utils[-1].voxelize_xyz(
                            self.xyz_camXs[batch_idx], self.Z, self.Y, self.X, return_vox_ind=False).unsqueeze(0)

                unpXs = self.vox_utils[-1].unproject_rgb_to_mem(
                            self.rgb_camXs[batch_idx], self.Z, self.Y, self.X, self.pix_T_cams[batch_idx]).unsqueeze(0)
            
            if use_enc3d:
                occ_halfmemX0_sup_, free_halfmemX0_sup_, _, _ = self.vox_utils[-1].prep_occs_supervision(
                    self.camX0s_T_camXs[batch_idx:batch_idx+1],
                    self.xyz_camXs[batch_idx:batch_idx+1],
                    self.Z4, self.Y4, self.X4,
                    agg=True
                    )
            else:
                occ_halfmemX0_sup_, free_halfmemX0_sup_, _, _ = self.vox_utils[-1].prep_occs_supervision(
                    camX0s_T_camXs,
                    xyz_camXs,
                    self.Z2, self.Y2, self.X2,
                    agg=True
                    )

            self.occXs.append(occXs)
            self.unpXs.append(unpXs)

            self.occ_halfmemX0_sup.append(occ_halfmemX0_sup_)
            self.free_halfmemX0_sup.append(free_halfmemX0_sup_)  


            if batch_idx==0 and self.summ_writer.save_this:

                occR0s = self.vox_utils[-1].voxelize_xyz(
                        self.xyz_camRs.reshape(self.B,-1,3), self.Z, self.Y, self.X, return_vox_ind=False)
                self.summ_writer.summ_occ('inputs/occR0s_input', occR0s, reduce_axes=[3])
                
                occX1_T_X0 = self.vox_utils[0].apply_4x4_to_vox(self.camX0s_T_camXs[0, 1:2], occXs[0, 1:2], binary_feat=True)
                occX0_T_X0 = self.vox_utils[0].apply_4x4_to_vox(self.camX0s_T_camXs[0, 0:1], occXs[0, 0:1], binary_feat=True)
                self.summ_writer.summ_occ('inputs/occX0_input', occXs[:,0], reduce_axes=[3])
                self.summ_writer.summ_occ('inputs/occX1_T_X0_input', occX1_T_X0, reduce_axes=[3])
                self.summ_writer.summ_occ('inputs/occX0_T_X0_input', occX0_T_X0, reduce_axes=[3])
                self.summ_writer.summ_occ('inputs/occX1_input', occXs[:,1], reduce_axes=[3])

        self.occXs = torch.cat(self.occXs, dim=0)
        self.unpXs = torch.cat(self.unpXs, dim=0)
        self.occ_halfmemX0_sup = torch.cat(self.occ_halfmemX0_sup, dim=0)
        self.free_halfmemX0_sup = torch.cat(self.free_halfmemX0_sup, dim=0)

        all_ok = True
        return all_ok

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        set_name = feed['set_name']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        self.summ_writer.summ_rgbs(f'inputs/rgbs', self.rgb_camXs.unbind(1))
        self.summ_writer.summ_unps('inputs/unps', self.unpXs.unbind(1), self.occXs.unbind(1))
        if self.summ_writer.save_this:
            # verify that inputs are aligned when warping
            vis_unpr_X1_T_X0 = self.vox_utils[0].apply_4x4_to_vox(self.camX0s_T_camXs[0, 1:2], self.unpXs[0, 1:2])
            vis_occ_X1_T_X0 = self.vox_utils[0].apply_4x4_to_vox(self.camX0s_T_camXs[0, 1:2], self.occXs[0, 1:2])
            self.summ_writer.summ_unp('inputs/unps_X1_T_X0', vis_unpr_X1_T_X0, vis_occ_X1_T_X0)
        
        if hyp.do_feat3d:
            assert(self.S==2)

            featX1_input = torch.cat([self.occXs[:,1:2], self.occXs[:,1:2]*self.unpXs[:,1:2]], dim=2)
            if use_enc3d:
                feat3d_loss, feat_halfmemX1, _ = self.feat3dnet(
                    __p(featX1_input),
                    summ_writer=self.summ_writer,
                    # set_name=set_name,
                )
            else:
                feat_halfmemX1, feat3d_loss = self.feat3dnet(
                    __p(featX0_input),
                    summ_writer=self.summ_writer,
                )

            # warp tp camX0 for loss
            feat_halfmemX0 = torch.cat([self.vox_utils[b].apply_4x4_to_vox(self.camX0s_T_camXs[b, 1:2], feat_halfmemX1[b:b+1]) for b in range(self.B)], dim=0)
            
            valid_halfmemX0 = torch.ones_like(feat_halfmemX0[:,0:1])
            valid_halfmemX0 = torch.cat([self.vox_utils[b].apply_4x4_to_vox(self.camX0s_T_camXs[b, 1:2], valid_halfmemX0[b:b+1]) for b in range(self.B)], dim=0)
            self.summ_writer.summ_feat(f'feat3d/feat_halfmemX0_from_X1', feat_halfmemX0, valid=valid_halfmemX0, pca=True)
            self.summ_writer.summ_oned(f'feat3d/valid_halfmemX0', valid_halfmemX0, bev=True, norm=False)

            total_loss += feat3d_loss
            
            if hyp.do_emb3d:
                feat_memX0_input = torch.cat([self.occXs[:,0:1], self.occXs[:,0:1]*self.unpXs[:,0:1]], dim=2)
                feat_memX0_input = __p(feat_memX0_input)
                if use_enc3d:
                    _, altfeat_halfmemX0, _ = self.feat3dnet_slow(feat_memX0_input)
                else:
                    altfeat_halfmemX0, _ = self.feat3dnet_slow(feat_memX0_input)
                altvalid_halfmemX0 = torch.ones_like(altfeat_halfmemX0[:,0:1])
                self.summ_writer.summ_feat(f'feat3d/altfeat_input', feat_memX0_input, pca=True)
                self.summ_writer.summ_feat(f'feat3d/altfeat_output', altfeat_halfmemX0, valid=altvalid_halfmemX0, pca=True)
                self.summ_writer.summ_oned(f'feat3d/altvalid_halfmemX0', altvalid_halfmemX0, bev=True, norm=False)

        if hyp.do_occ:
            
            # # be more conservative with "free"
            # weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            # self.free_halfmemX0_sup = 1.0 - (F.conv3d(1.0 - self.free_halfmemX0_sup, weights, padding=1)).clamp(0, 1)
            
            occ_loss, occ_memX0_pred = self.occnet(
                feat_halfmemX0, 
                occ_g=self.occ_halfmemX0_sup, 
                free_g=self.free_halfmemX0_sup, 
                valid=valid_halfmemX0.squeeze(1),
                summ_writer=self.summ_writer,
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
            total_loss += view_loss
        
        self.summ_writer.summ_scalar(f'loss', total_loss.cpu().item())

        # print("feat3d_loss", feat3d_loss)
        # print("occ_loss", occ_loss)
        # print("emb3d_loss", emb3d_loss)
        # print("total_loss", total_loss)

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
            # total_loss = torch.tensor(0.0).cuda()
            return None, None, False