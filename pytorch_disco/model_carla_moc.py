import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from backend import saverloader

from backend import inputs2 as inputs
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

np.set_printoptions(precision=2)
np.random.seed(0)

import ipdb
st = ipdb.set_trace

class CARLA_MOC(Model):
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

        set_nums = []
        set_names = []
        set_seqlens = []
        set_batch_sizes = []
        set_inputs = []
        set_writers = []
        set_log_freqs = []
        set_do_backprops = []
        set_dicts = []
        set_loaders = []

        for set_name in hyp.set_names:
            if hyp.sets_to_run[set_name]:
                set_nums.append(hyp.set_nums[set_name])
                set_names.append(set_name)
                set_seqlens.append(hyp.seqlens[set_name])
                set_batch_sizes.append(hyp.batch_sizes[set_name])
                set_inputs.append(self.all_inputs[set_name])
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=1000000, flush_secs=1000000))
                set_log_freqs.append(hyp.log_freqs[set_name])
                set_do_backprops.append(hyp.sets_to_backprop[set_name])
                set_dicts.append({})
                set_loaders.append(iter(set_inputs[-1]))

        for step in tqdm(range(self.start_iter+1, hyp.max_iters+1)):
            for i, (set_input) in enumerate(set_inputs):
                if step % len(set_input) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
                    set_loaders[i] = iter(set_input)

            for (set_num,
                 set_name,
                 set_seqlen,
                 set_batch_size,
                 set_input,
                 set_writer,
                 set_log_freq,
                 set_do_backprop,
                 set_dict,
                 set_loader
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
                set_loaders
            ):   

                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0

                output_dict = dict()

                if log_this or set_do_backprop:
                    # print('%s: set_num %d; log_this %d; set_do_backprop %d; ' % (set_name, set_num, log_this, set_do_backprop))
                    # print('log_this = %s' % log_this)
                    # print('set_do_backprop = %s' % set_do_backprop)
                          
                    read_start_time = time.time()

                    feed, _ = next(set_loader)
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

                    print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (hyp.name,
                                                                                        step,
                                                                                        hyp.max_iters,
                                                                                        total_time,
                                                                                        read_time,
                                                                                        iter_time,
                                                                                        loss_py,
                                                                                        set_name))
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
        
        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
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

        if feed['set_name']=='test':
            self.box_camRs = feed["box_traj_camR"]
            # box_camRs is B x S x 9
            self.score_s = feed["score_traj"]
            self.tid_s = torch.ones_like(self.score_s).long()
            self.lrt_camRs = utils.geom.convert_boxlist_to_lrtlist(self.box_camRs)
            self.lrt_camXs = utils.geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
            self.lrt_camX0s = utils.geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)
            self.lrt_camR0s = utils.geom.apply_4x4s_to_lrts(self.camR0s_T_camRs, self.lrt_camRs)

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

        if hyp.fit_vox:
            xyz_maxs = torch.max(self.xyz_camXs[:,1], dim=1)[0]
            xyz_mins = torch.min(self.xyz_camXs[:,1], dim=1)[0]

            # print(xyz_maxs)
            # print(xyz_mins)

            # shift_am = torch.tensor([(xyz_maxs[0][0] - torch.abs(xyz_mins[0][0]))/2., 0., 0.]).cuda().unsqueeze(0).unsqueeze(0)
            # xyz_camXs = xyz_camXs - shift_am
            # xyz_max = torch.max(xyz_camXs)
            scene_centroid_x_noise = np.random.normal(0, 0.2)
            scene_centroid_y_noise = np.random.normal(0, 0.2)
            scene_centroid_z_noise = np.random.normal(0, 0.2)
            xyz_max = torch.max(xyz_maxs, dim=1)[0]/2.
            self.scene_centroid = torch.tensor([0.0+scene_centroid_x_noise, 0.0+scene_centroid_y_noise, xyz_max+scene_centroid_z_noise]).unsqueeze(0).repeat(self.B,1).cuda()
            bounds = torch.tensor([-xyz_max, xyz_max, -xyz_max, xyz_max, -xyz_max, xyz_max]).cuda()
            self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, feed['set_name'], self.scene_centroid, bounds=bounds, assert_cube=True)
        else:
            if feed['set_name']=='test':
                # center on an object, so that it does not fall out of bounds
                scene_centroid = utils.geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0]
            else:
                # center randomly 
                scene_centroid_x_noise = np.random.normal(0, 0.2)
                scene_centroid_y_noise = np.random.normal(0, 0.2)
                scene_centroid_z_noise = np.random.normal(0, 0.2)
                sc_noise = np.array([scene_centroid_x_noise, scene_centroid_y_noise, scene_centroid_z_noise])
                # scene_centroid = torch.median(self.xyz_camXs[0,0], dim=0)[0] + torch.from_numpy(sc_noise).float().cuda()
                # scene_centroid = np.array([scene_centroid_x,
                #                            scene_centroid_y,
                #                            scene_centroid_z]).reshape([1, 3])
                # scene_centroid = scene_centroid.float().cuda().reshape([1, 3])
                # print(sc_noise)
                scene_centroid = torch.from_numpy(sc_noise).float().cuda().reshape([1, 3])

            self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, feed['set_name'], scene_centroid=scene_centroid, assert_cube=True)

        depth_camXs_, valid_camXs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        dense_xyz_camXs_ = utils.geom.depth2pointcloud(depth_camXs_, __p(self.pix_T_cams))
        dense_xyz_camRs_ = utils.geom.apply_4x4(__p(self.camRs_T_camXs), dense_xyz_camXs_)
        inbound_camXs_ = self.vox_util.get_inbounds(dense_xyz_camRs_, self.Z, self.Y, self.X).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        # depth_camXs = __u(depth_camXs_)
        self.valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)

        all_ok = True
        return all_ok

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        set_name = feed['set_name']
        print("SET_NAME:", set_name)
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        # since we are using multiview data, all Rs are aligned
        # we'll encode X0 with the fast net, then warp to R
        # we'll use the X0 version for occ loss, to get max labels
        # we'll encode X1/R1 with slow net, and use this for emb loss
        
        if hyp.do_feat3d:
            assert(self.S==2)
            # occ_memX0 = self.vox_util.voxelize_xyz(self.xyz_camXs[:,0], self.Z, self.Y, self.X)
            # unp_memX0 = self.vox_util.unproject_rgb_to_mem(
            #     self.rgb_camXs[:,0], self.Z, self.Y, self.X, self.pix_T_cams[:,0])
            # feat_memX0_input = torch.cat([occ_memX0, occ_memX0*unp_memX0], dim=1)
            # feat3d_loss, feat_halfmemX0, _ = self.feat3dnet(
            #     feat_memX0_input,
            #     self.summ_writer,
            # )
            # total_loss += feat3d_loss
            
            # valid_halfmemX0 = torch.ones_like(feat_halfmemX0[:,0:1])
            # # warp things to R0, for loss
            # feat_halfmemR = self.vox_util.apply_4x4_to_vox(self.camR0s_T_camXs[:, 0], feat_halfmemX0)
            # valid_halfmemR = self.vox_util.apply_4x4_to_vox(self.camR0s_T_camXs[:, 0], valid_halfmemX0)

            # occ_memX0_ = self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X)
            occXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
            # unp_memX0 = self.vox_util.unproject_rgb_to_mem(
            #     __p(self.rgb_camXs), self.Z, self.Y, self.X, self.pix_T_cams[:,0])
            unpXs = __u(self.vox_util.unproject_rgb_to_mem(
                __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
            # feat_memX0_input = torch.cat([occ_memX0, occ_memX0*unp_memX0], dim=1)
            featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)
            featXs_input_ = __p(featXs_input)
            feat3d_loss, feat_halfmemXs, _ = self.feat3dnet(
                featXs_input_,
                self.summ_writer,
            )
            total_loss += feat3d_loss
            
            feat_halfmemX0 = feat_halfmemXs[0:1]
            feat_halfmemX1 = feat_halfmemXs[1:2]
            valid_halfmemX0 = torch.ones_like(feat_halfmemX0[:,0:1])
            # warp things to R0, for loss
            feat_halfmemR = self.vox_util.apply_4x4_to_vox(self.camR0s_T_camXs[:, 0], feat_halfmemX0)
            valid_halfmemR = self.vox_util.apply_4x4_to_vox(self.camR0s_T_camXs[:, 0], valid_halfmemX0)

            # self.summ_writer.summ_feat('feat3d/feat_halfmemX1_input', feat_halfmemX1_input, pca=True)
            self.summ_writer.summ_feat(f'feat3d/{set_name}_feat_halfmemX0', feat_halfmemX0, valid=valid_halfmemX0, pca=True)
            self.summ_writer.summ_feat(f'feat3d/{set_name}_feat_halfmemR', feat_halfmemR, valid=valid_halfmemR, pca=True)
            self.summ_writer.summ_oned(f'feat3d/{set_name}_valid_halfmemR', valid_halfmemR, bev=True, norm=False)

            if hyp.do_emb3d:
                pixR_T_camX1 = utils.basic.matmul2(self.pix_T_cams[:,1], self.camRs_T_camXs[:,1])
                occ_memR = self.vox_util.voxelize_xyz(self.xyz_camRs[:,0], self.Z, self.Y, self.X)
                unp_memR = self.vox_util.unproject_rgb_to_mem(self.rgb_camXs[:,1], self.Z, self.Y, self.X, pixR_T_camX1)
                feat_memR_input = torch.cat([occ_memR, occ_memR*unp_memR], dim=1)
                _, altfeat_halfmemR, _ = self.feat3dnet_slow(feat_memR_input)
                altvalid_halfmemR = torch.ones_like(altfeat_halfmemR[:,0:1])
                self.summ_writer.summ_feat(f'feat3d/{set_name}_feat_memR_input', feat_memR_input, pca=True)
                self.summ_writer.summ_feat(f'feat3d/{set_name}_altfeat_halfmemR', altfeat_halfmemR, valid=altvalid_halfmemR, pca=True)
                self.summ_writer.summ_oned(f'feat3d/{set_name}_altvalid_halfmemR', altvalid_halfmemR, bev=True, norm=False)

        if hyp.do_occ:
            occ_halfmemX0_sup, free_halfmemX0_sup, _, _ = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                self.Z4, self.Y4, self.X4,
                agg=True)
            
            # be more conservative with "free"
            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            free_halfmemX0_sup = 1.0 - (F.conv3d(1.0 - free_halfmemX0_sup, weights, padding=1)).clamp(0, 1)

            occ_loss, occ_memX0_pred = self.occnet(
                feat_halfmemX0, 
                occ_g=occ_halfmemX0_sup,
                free_g=free_halfmemX0_sup,
                valid=valid_halfmemX0,
                summ_writer=self.summ_writer)
            total_loss += occ_loss

        # if hyp.do_rgb:
            
        #     pixX_T_camX0s = __u(
        #         utils.basic.matmul2(__p(self.pix_T_cams), __p(self.camXs_T_camX0s)))
        #     unp_halfmemX0s = __u(self.vox_util.unproject_rgb_to_mem(
        #             __p(self.rgb_camXs), self.Z2, self.Y2, self.X2, __p(pixX_T_camX0s)))
        #     occ_halfmemX0s = self.vox_util.voxelize_xyz(self.xyz_camX0s.squeeze(0), self.Z2, self.Y2, self.X2)
            
        #     rgb_agg_halfmemX0 = utils.basic.reduce_masked_mean(unp_halfmemX0s, occ_halfmemX0s.unsqueeze(0), dim=1)
        #     #occ_agg_halfmemX0 = torch.max(occ_halfmemX0s dim=1)[0]

        #     rgb_loss, rgb_e = self.rgbnet(
        #         feat_halfmemX0,
        #         rgb_g=rgb_agg_halfmemX0,
        #         valid=valid_halfmemX0,
        #         occ_e=F.sigmoid(occ_memX0_pred),
        #         occ_g=occ_halfmemX0_sup,
        #         summ_writer=self.summ_writer)
        #     total_loss += rgb_loss
            
        if hyp.do_emb3d:
            # compute 3D ML
            emb3d_loss = self.emb3dnet(
                feat_halfmemR,
                altfeat_halfmemR,
                valid_halfmemR.round(),
                altvalid_halfmemR.round(),
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
                f'{set_name}_rgb'
                )      
            # print(rgb_e.shape)              
            total_loss += view_loss

        self.summ_writer.summ_scalar(f'loss/{set_name}', total_loss.cpu().item())
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