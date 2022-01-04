import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from backend import saverloader

from torchvision.utils import make_grid, save_image

from torchvision.transforms import ToTensor, Resize, ToPILImage, CenterCrop

from backend import inputs2 as inputs

from model_base import Model
import sys
sys.path.append("torch-gqn")
from model import GQN
# from torch.distributions import Normal
import argparse

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

np.set_printoptions(precision=2)
np.random.seed(0)

import ipdb
st = ipdb.set_trace

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

device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"

class CARLA_GQN(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaGQNModel()
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
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=5e-4, betas=(0.9, 0.999), eps=1e-08)
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

        # for gqn
        sigma_i, sigma_f = 2.0, 0.7
        sigma = sigma_i

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

        for step in list(range(self.start_iter+1, hyp.max_iters+1)):
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
                    feed_cuda['sigma'] = sigma # used for GQN
                    
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
                    
                    sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - step/(2e5)), sigma_f)

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

            
class CarlaGQNModel(nn.Module):
    def __init__(self):
        super(CarlaGQNModel, self).__init__()

        # self.crop_guess = (18,18,18)
        # self.crop_guess = (2,2,2)
        self.crop = (18,18,18)
        
        if hyp.do_gqn:
            L =args.layers
            self.gqn = GQN(representation=hyp.gqn_representation, L=L, shared_core=args.shared_core).to(device)
            
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

    # def transform_viewpoint(self, v):
    #     w, z = torch.split(v, 3, dim=-1)
    #     y, p = torch.split(z, 1, dim=-1)

    #     # position, [yaw, pitch]
    #     view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    #     v_hat = torch.cat(view_vector, dim=-1)

    #     return v_hat
    
    def get_qgn_egomotion(self, origin_T_camXs):
        rot,trans = utils.geom.split_rt(origin_T_camXs.reshape(self.B*self.S,4,4))
        yaw, pitch, roll = utils.geom.rotm2eul(rot)
        view_vector = [trans, torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1), torch.cos(pitch).unsqueeze(1), torch.sin(pitch).unsqueeze(1)]
        v_hat = torch.cat(view_vector, dim=-1).reshape(self.B, self.S, 7)
        return v_hat

    
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
        # self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        # self.camRs_T_camR0 = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        # self.camRs_T_camXs = __u(torch.matmul(utils.geom.safe_inverse(__p(self.origin_T_camRs)), __p(self.origin_T_camXs)))
        # self.camXs_T_camRs = __u(utils.geom.safe_inverse(__p(self.camRs_T_camXs)))
        # self.camXs_T_camX0s = __u(utils.geom.safe_inverse(__p(self.camX0s_T_camXs)))
        # self.camX0_T_camR0 = utils.basic.matmul2(self.camX0s_T_camXs[:,0], self.camXs_T_camRs[:,0])
        # self.camR0s_T_camXs = utils.basic.matmul2(self.camR0s_T_camRs, self.camRs_T_camXs)
        self.sigma = feed['sigma']
        
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

        byte_to_tensor = lambda x: ToTensor()(Resize(64)(CenterCrop(min(self.H, self.W))((ToPILImage()(x+0.5))))).cuda()
        self.rgb_camXs = torch.stack([byte_to_tensor(frame) for frame in self.rgb_camXs.reshape(self.B*self.S,3,self.H,self.W)]).reshape(self.B,self.S,3,64,64)-0.5

        # self.xyz_camXs = feed["xyz_camXs"].float()
        # self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        # self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        # self.xyz_camR0s = __u(utils.geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))

        # if feed['set_name']=='test':
        #     self.box_camRs = feed["box_traj_camR"]
        #     # box_camRs is B x S x 9
        #     self.score_s = feed["score_traj"]
        #     self.tid_s = torch.ones_like(self.score_s).long()
        #     self.lrt_camRs = utils.geom.convert_boxlist_to_lrtlist(self.box_camRs)
        #     self.lrt_camXs = utils.geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
        #     self.lrt_camX0s = utils.geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)
        #     self.lrt_camR0s = utils.geom.apply_4x4s_to_lrts(self.camR0s_T_camRs, self.lrt_camRs)
        
        # if feed['set_name']=='test':
        #     # center on an object, so that it does not fall out of bounds
        #     scene_centroid = utils.geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0]
        # else:
        #     # center randomly 
        #     scene_centroid_x_noise = np.random.normal(0, 0.2)
        #     scene_centroid_y_noise = np.random.normal(0, 0.2)
        #     scene_centroid_z_noise = np.random.normal(0, 0.2)
        #     sc_noise = np.array([scene_centroid_x_noise, scene_centroid_y_noise, scene_centroid_z_noise])
        #     # scene_centroid = torch.median(self.xyz_camXs[0,0], dim=0)[0] + torch.from_numpy(sc_noise).float().cuda()
        #     # scene_centroid = np.array([scene_centroid_x,
        #     #                            scene_centroid_y,
        #     #                            scene_centroid_z]).reshape([1, 3])
        #     # scene_centroid = scene_centroid.float().cuda().reshape([1, 3])
        #     # print(sc_noise)
        #     scene_centroid = torch.from_numpy(sc_noise).float().cuda().reshape([1, 3])

        # self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, feed['set_name'], scene_centroid=scene_centroid, assert_cube=True)

        # depth_camXs_, valid_camXs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        # dense_xyz_camXs_ = utils.geom.depth2pointcloud(depth_camXs_, __p(self.pix_T_cams))
        # dense_xyz_camRs_ = utils.geom.apply_4x4(__p(self.camRs_T_camXs), dense_xyz_camXs_)
        # inbound_camXs_ = self.vox_util.get_inbounds(dense_xyz_camRs_, self.Z, self.Y, self.X).float()
        # inbound_camXs_ = torch.reshape(inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        # # depth_camXs = __u(depth_camXs_)
        # self.valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)

        all_ok = True
        return all_ok

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        set_name = feed['set_name']

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        # since we are using multiview data, all Rs are aligned
        # we'll encode X0 with the fast net, then warp to R
        # we'll use the X0 version for occ loss, to get max labels
        # we'll encode X1/R1 with slow net, and use this for emb loss
        
        if hyp.do_gqn:

            # self.qgn_v = self.get_qgn_egomotion(self.origin_T_camXs)

            
            # self.qgn_v = self.get_qgn_egomotion(self.origin_T_camXs)
            self.qgn_v = self.get_qgn_egomotion(self.camX0s_T_camXs) # encode relative

            # print(self.qgn_v)

            # x, v, x_q, v_q = sample_batch(x_data, v_data, D)
            elbo = self.gqn(self.rgb_camXs[:,0:1]+0.5, self.qgn_v[:,0:1], self.qgn_v[:,1], self.rgb_camXs[:,1]+0.5, self.sigma)

            # elbo_test = model(x_test, v_test, v_q_test, x_q_test, sigma)

            gqn_loss = -elbo.mean()
            total_loss += gqn_loss

            self.summ_writer.summ_scalar('gqn_loss', gqn_loss)

            # if len(args.device_ids)>1:
            #     kl_test = model.module.kl_divergence(x_test, v_test, v_q_test, x_q_test)
            #     x_q_rec_test = model.module.reconstruct(x_test, v_test, v_q_test, x_q_test)
            #     x_q_hat_test = model.module.generate(x_test, v_test, v_q_test)
            # else:
            if self.summ_writer.save_this:
                kl_test = self.gqn.kl_divergence(self.rgb_camXs[:,0:1]+0.5, self.qgn_v[:,0:1], self.qgn_v[:,1], self.rgb_camXs[:,1]+0.5)
                x_q_rec_test = self.gqn.reconstruct(self.rgb_camXs[:,0:1]+0.5, self.qgn_v[:,0:1], self.qgn_v[:,1], self.rgb_camXs[:,1]+0.5)
                x_q_hat_test = self.gqn.generate(self.rgb_camXs[:,0:1]+0.5, self.qgn_v[:,0:1], self.qgn_v[:,1])
                
                self.summ_writer.summ_scalar(f'{set_name}_kl', kl_test.mean())
                self.summ_writer.summ_rgb(f'view/{set_name}_ground_truth', self.rgb_camXs[:,1])
                self.summ_writer.summ_rgb(f'view/{set_name}_reconstruction', x_q_rec_test-0.5)
                self.summ_writer.summ_rgb(f'view/{set_name}_generation', x_q_hat_test-0.5)


        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
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
