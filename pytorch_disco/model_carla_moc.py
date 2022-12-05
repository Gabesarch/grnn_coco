import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from backend import saverloader

# if hyp.do_lescroart_moc:
#     from backend import inputs_lescroart as inputs
# elif hyp.do_carla_moc:
#     from backend import inputs2 as inputs
# else:
#     assert(False)

from backend import inputs2 as inputs
from backend import inputs_lescroart as inputs2

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


    def get_dataloader(
        self, 
        dataset_name, 
        trainset, 
        valset, 
        dataset_list_dir, 
        dataset_location, 
        dataset_filetype,
        trainset_format,
        trainset_seqlen,
        ):

        hyp.dataset_name = dataset_name
        hyp.trainset = trainset
        hyp.valset = valset
        hyp.dataset_list_dir = dataset_list_dir
        hyp.dataset_location = dataset_location
        hyp.dataset_filetype = dataset_filetype
        hyp.trainset_format = trainset_format
        hyp.trainset_seqlen = trainset_seqlen
        if hyp.trainset:
            hyp.name = "%s_%s" % (hyp.name, hyp.trainset)
            hyp.sets_to_run['train'] = True
        else:
            hyp.sets_to_run['train'] = False

        if hyp.valset:
            hyp.name = "%s_%s" % (hyp.name, hyp.valset)
            hyp.sets_to_run['val'] = True
        else:
            hyp.sets_to_run['val'] = False

        if hyp.testset:
            hyp.name = "%s_%s" % (hyp.name, hyp.testset)
            hyp.sets_to_run['test'] = True
        else:
            hyp.sets_to_run['test'] = False
        hyp.trainset_path = "%s/%s.txt" % (hyp.dataset_list_dir, hyp.trainset)
        hyp.valset_path = "%s/%s.txt" % (hyp.dataset_list_dir, hyp.valset)
        hyp.testset_path = "%s/%s.txt" % (hyp.dataset_list_dir, hyp.testset)
        hyp.data_paths['train'] = hyp.trainset_path
        hyp.data_paths['val'] = hyp.valset_path
        hyp.data_paths['test'] = hyp.testset_path
        hyp.data_formats['train'] = hyp.trainset_format
        hyp.data_formats['val'] = hyp.valset_format
        hyp.data_formats['test'] = hyp.testset_format
        if hyp.dataset_name=="markdata":
            all_inputs = inputs2.get_inputs()
        else:
            all_inputs = inputs.get_inputs()
        return all_inputs
            
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

        num_dataloaders = 1
        all_inputs = self.get_dataloader(
            hyp.dataset_name1, 
            hyp.trainset1, 
            hyp.valset1, 
            hyp.dataset_list_dir1, 
            hyp.dataset_location1, 
            hyp.dataset_filetype1,
            hyp.trainset_format1,
            hyp.trainset_seqlen1,
            )
        
        if hyp.dataset_name2 is not None:
            all_inputs2 = self.get_dataloader(
                hyp.dataset_name2, 
                hyp.trainset2, 
                hyp.valset2, 
                hyp.dataset_list_dir2, 
                hyp.dataset_location2, 
                hyp.dataset_filetype2,
                hyp.trainset_format2,
                hyp.trainset_seqlen2,
                )
            num_dataloaders += 1
        else:
            all_inputs2 = {set_name:[] for set_name in hyp.set_names}
        
        if hyp.dataset_name3 is not None:
            all_inputs3 = self.get_dataloader(
                hyp.dataset_name3, 
                hyp.trainset3, 
                hyp.valset3, 
                hyp.dataset_list_dir3, 
                hyp.dataset_location3, 
                hyp.dataset_filetype3,
                hyp.trainset_format3,
                hyp.trainset_seqlen3,
                )
            num_dataloaders += 1
        else:
            all_inputs3 = {set_name:[] for set_name in hyp.set_names}
        
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
                if set_name in ["train"]:
                    set_inputs2.append(all_inputs2[set_name])
                    set_inputs3.append(all_inputs3[set_name])
                else:
                    set_inputs2.append(None)
                    set_inputs3.append(None)
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=1000000, flush_secs=1000000))
                set_log_freqs.append(hyp.log_freqs[set_name])
                set_do_backprops.append(hyp.sets_to_backprop[set_name])
                set_dicts.append({})
                set_loaders.append(iter(set_inputs[-1]))
                if set_name in ["train"]:
                    set_loaders2.append(iter(set_inputs2[-1]))
                    set_loaders3.append(iter(set_inputs3[-1]))
                else:
                    set_loaders2.append(None)
                    set_loaders3.append(None)

        for step in tqdm(range(self.start_iter+1, hyp.max_iters+1)):
            # for i, (set_input) in enumerate(set_inputs):
            #     if step % len(set_input) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
            #         set_loaders[i] = iter(set_input)

            # for i, (set_input2) in enumerate(set_inputs2):
            #     if step % len(set_input2) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
            #         set_loaders2[i] = iter(set_input2)

            # for i, (set_input3) in enumerate(set_inputs3):
            #     if step % len(set_input3) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
            #         set_loaders3[i] = iter(set_input3)

            for (set_num,
                 set_name,
                 set_seqlen,
                 set_batch_size,
                 set_input,
                 set_input2,
                 set_input3,
                 set_writer,
                 set_log_freq,
                 set_do_backprop,
                 set_dict,
                 set_loader,
                 set_loader2,
                 set_loader3,
            ) in zip(
                set_nums,
                set_names,
                set_seqlens,
                set_batch_sizes,
                set_inputs,
                set_inputs2,
                set_inputs3,
                set_writers,
                set_log_freqs,
                set_do_backprops,
                set_dicts,
                set_loaders,
                set_loaders2,
                set_loaders3,
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

                    dataloader_index = step % num_dataloaders
                    if set_name in ["val", "test"]:
                        dataloader_index = 0
                    if dataloader_index==0:
                        try:
                            feed = next(set_loader)
                        except StopIteration:
                            for i, (set_input) in enumerate(set_inputs):
                                #restart after one epoch. Note this does nothing for the tfrecord loader
                                set_loaders[i] = iter(set_input)
                            continue
                            # feed = next(set_loader)
                    elif dataloader_index==1:
                        try:
                            feed = next(set_loader2)
                        except StopIteration:
                            for i, (set_input2) in enumerate(set_inputs2):
                                if i>0:
                                    continue
                                #restart after one epoch. Note this does nothing for the tfrecord loader
                                set_loaders2[i] = iter(set_input2)
                            # set_loaders2[0] = iter(set_input2)
                            continue
                        
                            # feed = next(set_loader2)
                    elif dataloader_index==2:   
                        try:
                            feed = next(set_loader3)
                        except StopIteration:
                            for i, (set_input3) in enumerate(set_inputs3):
                                if i>0:
                                    continue
                                #restart after one epoch. Note this does nothing for the tfrecord loader
                                set_loaders3[i] = iter(set_input3)
                            # set_loaders3[0] = iter(set_input3)
                            continue
                            # feed = next(set_loader3)
                            
                    if len(feed)==2:
                        feed = feed[0]

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
            self.vox_utils = []
            # if "depth_at_center" in feed.keys():
            #     offset_x, offset_y, offset_z = feed["depth_at_center"].cpu().numpy()
            # else:
            #     offset_x, offset_y, offset_z = 0., 0., 0. 
            for b in range(self.B):
                
                xyz_maxs = torch.max(self.xyz_camXs[b:b+1,1], dim=1)[0]
                xyz_mins = torch.min(self.xyz_camXs[b:b+1,1], dim=1)[0]

                # print(xyz_maxs)
                # print(xyz_mins)

                # shift_am = torch.tensor([(xyz_maxs[0][0] - torch.abs(xyz_mins[0][0]))/2., 0., 0.]).cuda().unsqueeze(0).unsqueeze(0)
                # xyz_camXs = xyz_camXs - shift_am
                # xyz_max = torch.max(xyz_camXs)
                scene_centroid_x_noise = np.random.normal(0, 0.2) #+ offset_x
                scene_centroid_y_noise = np.random.normal(0, 0.2) #+ offset_y
                scene_centroid_z_noise = np.random.normal(0, 0.2) #+ offset_z
                if "depth_at_center" in feed.keys():
                    depth_at_center, center_to_boundary = (feed["depth_at_center"], feed["center_to_boundary"])
                    scene_centroid_B = torch.tensor([0.0+scene_centroid_x_noise, 0.0+scene_centroid_y_noise, depth_at_center[b]+scene_centroid_z_noise]).unsqueeze(0).repeat(self.B,1).cuda()
                    center_to_boundary = center_to_boundary[b] #torch.max(center_to_boundary)
                    bounds_B = torch.tensor([-center_to_boundary, center_to_boundary, -center_to_boundary, center_to_boundary, -center_to_boundary, center_to_boundary]).cuda()
                else:
                    xyz_max = torch.clip(torch.max(xyz_maxs, dim=1)[0], max=20.0) / 2.
                    scene_centroid_B = torch.tensor([np.zeros(1)+scene_centroid_x_noise, np.zeros(1)+scene_centroid_y_noise, xyz_max+scene_centroid_z_noise]).cuda().squeeze(0).reshape(1,3)
                    # scene_centroid_B = torch.tensor([np.zeros(1)+scene_centroid_x_noise, np.zeros(1)+scene_centroid_y_noise, xyz_max+scene_centroid_z_noise]).cuda().squeeze(0).reshape(1,3)
                    bounds_B = torch.stack([-xyz_max, xyz_max, -xyz_max, xyz_max, -xyz_max, xyz_max]).cuda()

                self.vox_utils.append(utils.vox.Vox_util(self.Z, self.Y, self.X, feed['set_name'], scene_centroid_B, bounds=bounds_B, assert_cube=True))
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

        dense_xyz_camRs_ = dense_xyz_camRs_.reshape(self.B, self.S, dense_xyz_camRs_.shape[1], dense_xyz_camRs_.shape[2])
        inbound_camXs__ = []
        for b in range(self.B):
            inbound_camXs_ = self.vox_utils[b].get_inbounds(dense_xyz_camRs_[b], self.Z, self.Y, self.X).float()
            inbound_camXs__.append(inbound_camXs_)
        inbound_camXs_ = torch.stack(inbound_camXs__)
        inbound_camXs_ = inbound_camXs_.reshape(self.B*self.S, inbound_camXs_.shape[2])
        inbound_camXs_ = torch.reshape(inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        # depth_camXs = __u(depth_camXs_)
        self.valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)

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
            occXs_ = []
            for b in range(self.B):
                occXs = self.vox_utils[b].voxelize_xyz(self.xyz_camXs[b], self.Z, self.Y, self.X)
                occXs_.append(occXs)
            occXs = torch.stack(occXs_)
            # unp_memX0 = self.vox_util.unproject_rgb_to_mem(
            #     __p(self.rgb_camXs), self.Z, self.Y, self.X, self.pix_T_cams[:,0])
            unpXs_ = []
            for b in range(self.B):
                unpXs = self.vox_utils[b].unproject_rgb_to_mem(
                    self.rgb_camXs[b], self.Z, self.Y, self.X, self.pix_T_cams[b])
                unpXs_.append(unpXs)
            unpXs = torch.stack(unpXs_)
            
            # feat_memX0_input = torch.cat([occ_memX0, occ_memX0*unp_memX0], dim=1)
            featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)
            featXs_input_ = __p(featXs_input)
            feat3d_loss, feat_halfmemXs, _ = self.feat3dnet(
                featXs_input_,
                summ_writer=self.summ_writer,
                # set_name=set_name,
            )
            total_loss += feat3d_loss

            feat_halfmemXs = feat_halfmemXs.reshape(self.B, self.S, feat_halfmemXs.shape[1], self.Z4, self.Y4, self.X4)
            
            feat_halfmemX0 = feat_halfmemXs[:, 0:1]
            feat_halfmemX1 = feat_halfmemXs[:, 1:2]
            valid_halfmemX0 = torch.ones_like(feat_halfmemX0[:,:,0:1])

            # warp things to R0, for loss
            feat_halfmemR = []
            valid_halfmemR = []
            for b in range(self.B):
                feat_halfmemR.append(self.vox_utils[b].apply_4x4_to_vox(self.camR0s_T_camXs[b, 0:1], feat_halfmemX0[b]))
                valid_halfmemR.append(self.vox_utils[b].apply_4x4_to_vox(self.camR0s_T_camXs[b, 0:1], valid_halfmemX0[b]))
            feat_halfmemR = torch.stack(feat_halfmemR)
            valid_halfmemR = torch.stack(valid_halfmemR)

            # # warp things to X0, for loss
            # feat_halfmemX0 = []
            # valid_halfmemX0 = []
            # for b in range(self.B):
            #     feat_halfmemX0.append(self.vox_utils[b].apply_4x4_to_vox(self.camX0s_T_camXs[b, 0:1], feat_halfmemX0[b]))
            #     valid_halfmemX0.append(self.vox_utils[b].apply_4x4_to_vox(self.camX0s_T_camXs[b, 0:1], valid_halfmemX0[b]))
            # feat_halfmemX0 = torch.stack(feat_halfmemX0)
            # valid_halfmemX0 = torch.stack(valid_halfmemX0)

            # self.summ_writer.summ_feat('feat3d/feat_halfmemX1_input', feat_halfmemX1_input, pca=True)
            self.summ_writer.summ_feat(f'feat3d/feat_halfmemX0', feat_halfmemX0[0], valid=valid_halfmemX0[0], pca=True)
            self.summ_writer.summ_feat(f'feat3d/feat_halfmemR', feat_halfmemR[0], valid=valid_halfmemR[0], pca=True)
            self.summ_writer.summ_oned(f'feat3d/valid_halfmemR', valid_halfmemR[0], bev=True, norm=False)

            if hyp.do_emb3d:

                if use_camR:
                    pixR_T_camX1 = utils.basic.matmul2(self.pix_T_cams[:,1], self.camRs_T_camXs[:,1])
                    occ_memR = []
                    unp_memR = []
                    for b in range(self.B):
                        occ_memR.append(self.vox_utils[b].voxelize_xyz(self.xyz_camRs[b:b+1,0], self.Z, self.Y, self.X))
                        unp_memR.append(self.vox_utils[b].unproject_rgb_to_mem(self.rgb_camXs[b:b+1,1], self.Z, self.Y, self.X, pixR_T_camX1[b:b+1]))
                    occ_memR = __p(torch.stack(occ_memR))
                    unp_memR = __p(torch.stack(unp_memR))
                    feat_memR_input = torch.cat([occ_memR, occ_memR*unp_memR], dim=1)
                    _, altfeat_halfmemR, _ = self.feat3dnet_slow(feat_memR_input)
                    altvalid_halfmemR = torch.ones_like(altfeat_halfmemR[:,0:1])
                    self.summ_writer.summ_feat(f'feat3d/feat_memR_input', feat_memR_input, pca=True)
                    self.summ_writer.summ_feat(f'feat3d/altfeat_halfmemR', altfeat_halfmemR, valid=altvalid_halfmemR, pca=True)
                    self.summ_writer.summ_oned(f'feat3d/altvalid_halfmemR', altvalid_halfmemR, bev=True, norm=False)
                else:
                    pixX0_T_camX1 = utils.basic.matmul2(self.pix_T_cams[:,1], self.camX0s_T_camXs[:,1])
                    occ_memX0 = []
                    unp_memX0 = []
                    for b in range(self.B):
                        occ_memX0.append(self.vox_utils[b].voxelize_xyz(self.xyz_camX0s[b:b+1,1], self.Z, self.Y, self.X))
                        unp_memX0.append(self.vox_utils[b].unproject_rgb_to_mem(self.rgb_camXs[b:b+1,1], self.Z, self.Y, self.X, pixX0_T_camX1[b:b+1]))
                    occ_memX0 = __p(torch.stack(occ_memX0))
                    unp_memX0 = __p(torch.stack(unp_memX0))
                    feat_memX0_input = torch.cat([occ_memX0, occ_memX0*unp_memX0], dim=1)
                    _, altfeat_halfmemX0, _ = self.feat3dnet_slow(feat_memX0_input)
                    altvalid_halfmemX0 = torch.ones_like(altfeat_halfmemX0[:,0:1])
                    self.summ_writer.summ_feat(f'feat3d/feat_memX0_input', feat_memX0_input, pca=True)
                    self.summ_writer.summ_feat(f'feat3d/altfeat_halfmemX0', altfeat_halfmemX0, valid=altvalid_halfmemX0, pca=True)
                    self.summ_writer.summ_oned(f'feat3d/altvalid_halfmemX0', altvalid_halfmemX0, bev=True, norm=False)

        if hyp.do_occ:
            occ_halfmemX0_sup = []
            free_halfmemX0_sup = []
            for b in range(self.B):
                occ_halfmemX0_sup_, free_halfmemX0_sup_, _, _ = self.vox_utils[b].prep_occs_supervision(
                    self.camX0s_T_camXs[b:b+1],
                    self.xyz_camXs[b:b+1],
                    self.Z4, self.Y4, self.X4,
                    agg=True)
                occ_halfmemX0_sup.append(occ_halfmemX0_sup_)
                free_halfmemX0_sup.append(free_halfmemX0_sup_)
            occ_halfmemX0_sup = __p(torch.stack(occ_halfmemX0_sup))
            free_halfmemX0_sup = __p(torch.stack(free_halfmemX0_sup))
            
            # be more conservative with "free"
            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            free_halfmemX0_sup = 1.0 - (F.conv3d(1.0 - free_halfmemX0_sup, weights, padding=1)).clamp(0, 1)
            
            occ_loss, occ_memX0_pred = self.occnet(
                utils.basic.pack_seqdim(feat_halfmemX0, self.B), 
                occ_g=occ_halfmemX0_sup, #utils.basic.pack_seqdim(occ_halfmemX0_sup, self.B),
                free_g=free_halfmemX0_sup, #utils.basic.pack_seqdim(free_halfmemX0_sup, self.B),
                valid=utils.basic.pack_seqdim(valid_halfmemX0, self.B),
                summ_writer=self.summ_writer,
                # set_name=set_name,
                )
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
            if use_camR:
                # compute 3D ML
                emb3d_loss = self.emb3dnet(
                    utils.basic.pack_seqdim(feat_halfmemR, self.B),
                    altfeat_halfmemR,
                    utils.basic.pack_seqdim(valid_halfmemR, self.B).round(),
                    altvalid_halfmemR.round(),
                    self.summ_writer)
            else:
                emb3d_loss = self.emb3dnet(
                    utils.basic.pack_seqdim(feat_halfmemX0, self.B),
                    altfeat_halfmemX0,
                    utils.basic.pack_seqdim(valid_halfmemX0, self.B).round(),
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