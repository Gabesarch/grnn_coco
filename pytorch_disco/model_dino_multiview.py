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

import sys
sys.path.append("dino")
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from PIL import Image
import vision_transformer as vits
from vision_transformer import DINOHead
import dino_utils as utils_dino

from tqdm import tqdm

# np.set_printoptions(precision=2)
# np.random.seed(0)

import ipdb
st = ipdb.set_trace



class DINO_MULTIVIEW(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")

        hyp.arch = hyp.arch.replace("deit", "vit")
        # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
        if hyp.arch in vits.__dict__.keys():
            student = vits.__dict__[hyp.arch](
                patch_size=hyp.patch_size,
                drop_path_rate=hyp.drop_path_rate,  # stochastic depth
            )
            teacher = vits.__dict__[hyp.arch](patch_size=hyp.patch_size)
            embed_dim = student.embed_dim
        # if the network is a XCiT
        elif hyp.arch in torch.hub.list("facebookresearch/xcit"):
            student = torch.hub.load('facebookresearch/xcit', hyp.arch,
                                    pretrained=False, drop_path_rate=hyp.drop_path_rate)
            teacher = torch.hub.load('facebookresearch/xcit', hyp.arch, pretrained=False)
            embed_dim = student.embed_dim
        # otherwise, we check if the architecture is in torchvision models
        elif hyp.arch in torchvision_models.__dict__.keys():
            student = torchvision_models.__dict__[hyp.arch]()
            teacher = torchvision_models.__dict__[hyp.arch]()
            embed_dim = student.fc.weight.shape[1]
        else:
            print(f"Unknow architecture: {hyp.arch}")

        # multi-crop wrapper handles forward with inputs of different resolutions
        student = utils_dino.MultiCropWrapper(student, DINOHead(
            embed_dim,
            hyp.out_dim,
            use_bn=hyp.use_bn_in_head,
            norm_last_layer=hyp.norm_last_layer,
        ))
        teacher = utils_dino.MultiCropWrapper(
            teacher,
            DINOHead(embed_dim, hyp.out_dim, hyp.use_bn_in_head),
        )
        # move networks to gpu
        student, teacher = student.cuda(), teacher.cuda()
        # teacher = nn.parallel.DataParallel(teacher)
        # synchronize batch norms (if any)
        if utils_dino.has_batchnorms(student):
            # student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            # teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

            # we need DDP wrapper to have synchro batch norms working...
            # teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[hyp.gpu])
            if hyp.dataparallel:
                teacher = nn.parallel.DataParallel(teacher)
            teacher_without_ddp = teacher.module
        else:
            # teacher_without_ddp and teacher are the same thing
            # teacher = nn.parallel.DataParallel(teacher)
            teacher_without_ddp = teacher#.module
            # teacher_without_ddp = teacher
        # student = nn.parallel.DistributedDataParallel(student, device_ids=[hyp.gpu])
        if hyp.dataparallel:
            student = nn.parallel.DataParallel(student)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # teacher and student start with the same weights
        if hyp.dataparallel:
            teacher_without_ddp.load_state_dict(student.module.state_dict())
        else:
            teacher_without_ddp.load_state_dict(student.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"Student and Teacher are built: they are both {hyp.arch} network.")

        # ============ preparing loss ... ============
        dino_loss = DINOLoss(
            hyp.out_dim,
            hyp.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            hyp.warmup_teacher_temp,
            hyp.teacher_temp,
            hyp.warmup_teacher_temp_epochs,
            hyp.epochs,
        ).cuda()

        # momentum_schedule = utils.cosine_scheduler(hyp.momentum_teacher, 1,
        #                                        hyp.epochs, 5000)
        # st()

        # ============ preparing optimizer ... ============
        params_groups = utils_dino.get_params_groups(student)
        # if hyp.optimizer == "adamw":
        #     optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        # elif hyp.optimizer == "sgd":
        #     optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        # elif hyp.optimizer == "lars":
        #     optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

        if hyp.lr > 0:
            # params_to_optimize = self.model.parameters()
            self.optimizer = torch.optim.Adam(params_groups, lr=hyp.lr)
        else:
            self.optimizer = None

        # self.student = student
        # self.teacher = teacher
        # self.teacher_without_ddp = teacher_without_ddp

        self.model = CarlaMocModel(student, teacher, teacher_without_ddp, dino_loss)

        # # for mixed precision training
        # fp16_scaler = None
        # if hyp.use_fp16:
        #     fp16_scaler = torch.cuda.amp.GradScaler()

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
        valset_format,
        valset_seqlen,
        ):

        hyp.dataset_name = dataset_name
        hyp.trainset = trainset
        hyp.valset = valset
        hyp.dataset_list_dir = dataset_list_dir
        hyp.dataset_location = dataset_location
        hyp.dataset_filetype = dataset_filetype
        hyp.trainset_format = trainset_format
        hyp.trainset_seqlen = trainset_seqlen
        hyp.valset_format = valset_format
        hyp.valset_seqlen = valset_seqlen
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

    def save_images(self, dataset_path, split):
        import os
        print('dataset_path = %s' % dataset_path)
        with open(dataset_path) as f:
            content = f.readlines()
        dataset_location = dataset_path.split('/')[:-1]
        dataset_location = '/'.join(dataset_location)
        print('dataset_loc = %s' % dataset_location)
        
        records = [hyp.dataset_location + '/' + line.strip() for line in content]
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))

        for filename in records:
            print(f"processing {filename}")
            d = np.load(filename, allow_pickle=True)
            if 'rgb_camXs_raw' in d:
                rgb_camXs = d['rgb_camXs_raw']
            else:
                rgb_camXs = d['rgb_camXs']            
            for s in range(len(rgb_camXs)):
                img = Image.fromarray(rgb_camXs[s])
                path_new = f"{filename.rsplit('.', 1)[0]}_{s}.jpeg"
                if not os.path.exists(path_new):
                    print(f"processing {s}")
                    print(path_new)
                    img.save(path_new)

            
    # override go from base
    def go(self):
        self.start_time = time.time()
        self.initialize_model()
        print("------ Done creating models ------")

        # if hyp.lr > 0:
        #     params_to_optimize = self.model.parameters()
        #     self.optimizer = torch.optim.Adam(params_to_optimize, lr=hyp.lr)
        # else:
        #     self.optimizer = None
        
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
            hyp.valset_format1,
            hyp.valset_seqlen1,
            )
        # self.save_images(hyp.trainset_path, "train")
        # self.save_images(hyp.valset_path, "val")
        
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
                "",
                "",
                )
            num_dataloaders += 1
            # self.save_images(hyp.trainset_path, "train")
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
                "",
                "",
                )
            num_dataloaders += 1
            # self.save_images(hyp.trainset_path, "train")
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
                        
                        # self.optimizer.zero_grad()
                        # loss.backward()
                        # self.optimizer.step()
                        self.optimizer.zero_grad()
                        param_norms = None
                        loss.backward()
                        # if args.clip_grad:
                        #     param_norms = utils.clip_gradients(student, args.clip_grad)
                        # utils.cancel_gradients_last_layer(epoch, student,
                        #                                 args.freeze_last_layer)
                        self.optimizer.step()

                    # EMA update for the teacher
                    if hyp.dataparallel:
                        with torch.no_grad():
                            m = 0.999 #momentum_schedule[step]  # momentum parameter
                            for param_q, param_k in zip(self.model.student.module.parameters(), self.model.teacher_without_ddp.parameters()):
                                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                    else:
                        with torch.no_grad():
                            m = 0.999 #momentum_schedule[step]  # momentum parameter
                            for param_q, param_k in zip(self.model.student.parameters(), self.model.teacher_without_ddp.parameters()):
                                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                    # if hyp.do_emb3d:
                    #     def update_slow_network(slow_net, fast_net, beta=0.999):
                    #         param_k = slow_net.state_dict()
                    #         param_q = fast_net.named_parameters()
                    #         for n, q in param_q:
                    #             if n in param_k:
                    #                 param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
                    #         slow_net.load_state_dict(param_k)
                    #     update_slow_network(self.model.feat3dnet_slow, self.model.feat3dnet)
                        
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
    def __init__(self, student, teacher, teacher_without_ddp, dino_loss):
        super(CarlaMocModel, self).__init__()

        self.student = student
        self.teacher = teacher
        self.teacher_without_ddp = teacher_without_ddp
        self.dino_loss = dino_loss
        self.data_aug = DataAugmentationDINO(
            hyp.global_crops_scale,
            hyp.local_crops_scale,
            hyp.local_crops_number,
        )

        self.invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
            
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
        
        self.images = feed["rgb_camXs"]
        self.images = [im.cuda(non_blocking=True) for im in self.images]

        all_ok = True
        return all_ok

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        set_name = feed['set_name']
        # print("SET_NAME:", set_name)
        # total_loss = torch.tensor(0.0).cuda()
        # print(len(self.images), global_step)
        teacher_output = self.teacher(self.images[:2])  # only the 2 global views pass through the teacher
        student_output = self.student(self.images)
        loss = self.dino_loss(student_output, teacher_output, 100)

        # print(len(self.images))

        if self.summ_writer.save_this:
            self.summ_writer.summ_rgb(f'inputs/global1', self.invTrans(self.images[0][0:1])-0.5)
            self.summ_writer.summ_rgb(f'inputs/global2', self.invTrans(self.images[1][0:1])-0.5)
            self.summ_writer.summ_rgb(f'inputs/local1', self.invTrans(self.images[2][0:1])-0.5)
            self.summ_writer.summ_rgb(f'inputs/local2', self.invTrans(self.images[3][0:1])-0.5)
            self.summ_writer.summ_rgb(f'inputs/local3', self.invTrans(self.images[4][0:1])-0.5)
            self.summ_writer.summ_rgb(f'inputs/local4', self.invTrans(self.images[5][0:1])-0.5)
            self.summ_writer.summ_rgb(f'inputs/local5', self.invTrans(self.images[6][0:1])-0.5)
            self.summ_writer.summ_rgb(f'inputs/local6', self.invTrans(self.images[7][0:1])-0.5)
            self.summ_writer.summ_rgb(f'inputs/local7', self.invTrans(self.images[8][0:1])-0.5)
            self.summ_writer.summ_rgb(f'inputs/local8', self.invTrans(self.images[9][0:1])-0.5)

        self.summ_writer.summ_scalar(f'loss', loss.cpu().item())

        return loss, results, False

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

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                try:
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                except:
                    st()
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        # with Data
        batch_center = batch_center / (len(teacher_output) * 1) #dist.get_world_size())
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils_dino.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils_dino.GaussianBlur(0.1),
            utils_dino.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils_dino.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops