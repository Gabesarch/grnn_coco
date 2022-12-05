# import tensorflow as tf
import numpy as np
import scipy
import torch
import pickle
from torch.utils.data import DataLoader
import hyperparams as hyp
import utils.py
from backend import readers
import os, json, random, imageio
import utils.improc
np.set_printoptions(precision=2, suppress=True)
from torchvision import datasets, transforms
from PIL import Image
import dino.dino_utils as utils_dino
import time
from os import listdir
from os.path import isfile, join

import ipdb
st = ipdb.set_trace

class IndexedDataset(torch.utils.data.Dataset):
    """ 
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset
        
    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # img, label = self.base[idx]
        # return (img, label, idx)
        feed = self.base[idx]
        return (feed, idx)

class NpzDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, shuffle, data_format, data_consec, seqlen):
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
 
        self.records = records
        self.shuffle = shuffle
        self.data_format = data_format
        self.data_consec = data_consec
        self.seqlen = seqlen

        if self.data_format=="dino_multiview":
            self.data_aug = DataAugmentationDINO(
                hyp.global_crops_scale,
                hyp.local_crops_scale,
                hyp.local_crops_number,
            )

    def __getitem__(self, index):
        # print(self.data_format)
        if self.data_format=='dino_multiview':
            time1 = time.time()
            filename = self.records[index]
            filename = filename.rsplit('.', 1)[0]
            folder, file_start = filename.rsplit('/', 1)
            onlyfiles = [f for f in listdir(folder) if (isfile(join(folder, f)) and file_start in f and '.jpeg' in f)]
            inds = np.random.randint(0, len(onlyfiles), self.seqlen)
            chosen = [onlyfiles[idx] for idx in list(inds)]
            if hyp.predict_view1_only:
                view1 = image = Image.open(os.path.join(folder, chosen[0])) #Image.fromarray(rgb_camXs[0])
                view2 = view1
            else:
                assert(len(chosen)==2)
                view1 = Image.open(os.path.join(folder, chosen[0]))
                view2 = Image.open(os.path.join(folder, chosen[1]))
            rgb = self.data_aug(view1, view2)
            d = {}
            d['rgb_camXs'] = rgb
            return d
        elif (self.data_format=='seq' or
            self.data_format=='multiview' or
            self.data_format=='nuscenes' or
            self.data_format=='ktrack' or
            self.data_format=='kodo' or
            self.data_format=='traj' or
            self.data_format=='complete' or
            self.data_format=='oldtraj' or
            self.data_format=='simpletraj'):
            filename = self.records[index]
            # time1 = time.time()
            d = np.load(filename, allow_pickle=True)
            # time2 = time.time()
            # print(1, time2-time1)
            d = dict(d)
            # elif self.data_format=='nuscenes':
            #     filename = self.records[index]
            #     d = np.load(filename, allow_pickle=True)['data']
            #     d = d[()]
        else:
            assert(False) # reader not ready yet

        # if the sequence length > 2, select S frames
        if self.shuffle:
            # time1 = time.time()
            d = self.random_select_single(d, num_samples=self.seqlen)
            # time2 = time.time()
            # print(2, time2-time1)
        else:
            d = self.non_random_select_single(d, num_samples=self.seqlen)

        if hyp.do_time_flip:
            d = self.random_time_flip_single(d)

        if self.data_format=='simpletraj':
            # for k, v in d.items():
            #     print(k)
            rgb_camX0 = d['rgb_camX0']
            # move channel dim inward, like pytorch wants
            rgb_camX0 = np.transpose(rgb_camX0, axes=[2, 0, 1])
            rgb_camX0 = utils.py.preprocess_color(rgb_camX0)
            d['rgb_camX0'] = rgb_camX0
        elif self.data_format=='nuscenes':
            # for k, v in d.items():
            #     print(k)
            #     print(v.shape)
            #     # print(k, v.shape)
            # print('rgb_camXs', rgb_camXs.shape)
            
            # move channel dim inward, like pytorch wants
            rgb_camXs = d['rgb_camXs']
            rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
            rgb_camXs = utils.py.preprocess_color(rgb_camXs)
            d['rgb_camXs'] = rgb_camXs
            pass
        elif self.data_format=='complete':
            rgb_cam0s = d['rgb_cam0s']
            # move channel dim inward, like pytorch wants
            rgb_cam0s = np.transpose(rgb_cam0s, axes=[0, 3, 1, 2])
            rgb_cam0s = utils.py.preprocess_color(rgb_cam0s)
            d['rgb_cam0s'] = rgb_cam0s

            rgb_cam1s = d['rgb_cam1s']
            # move channel dim inward, like pytorch wants
            rgb_cam1s = np.transpose(rgb_cam1s, axes=[0, 3, 1, 2])
            rgb_cam1s = utils.py.preprocess_color(rgb_cam1s)
            d['rgb_cam1s'] = rgb_cam1s
            
            rgb_cam2s = d['rgb_cam2s']
            # move channel dim inward, like pytorch wants
            rgb_cam2s = np.transpose(rgb_cam2s, axes=[0, 3, 1, 2])
            rgb_cam2s = utils.py.preprocess_color(rgb_cam2s)
            d['rgb_cam2s'] = rgb_cam2s
            
            rgb_cam3s = d['rgb_cam3s']
            # move channel dim inward, like pytorch wants
            rgb_cam3s = np.transpose(rgb_cam3s, axes=[0, 3, 1, 2])
            rgb_cam3s = utils.py.preprocess_color(rgb_cam3s)
            d['rgb_cam3s'] = rgb_cam3s
            
            rgb_camRs = d['rgb_camRs']
            # move channel dim inward, like pytorch wants
            rgb_camRs = np.transpose(rgb_camRs, axes=[0, 3, 1, 2])
            rgb_camRs = utils.py.preprocess_color(rgb_camRs)
            d['rgb_camRs'] = rgb_camRs
        elif self.data_format=='dino_multiview':
            # print("HERE")
            if 'rgb_camXs_raw' in d:
                rgb_camXs = d['rgb_camXs_raw']
            else:
                rgb_camXs = d['rgb_camXs']
            # move channel dim inward, like pytorch wants
            # rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
            # time2 = time.time()
            # print(2, time2-time1)
            # time1 = time.time()
            if hyp.predict_view1_only:
                # print('here')
                view1 = Image.fromarray(rgb_camXs[0])
                view2 = view1
            else:
                assert(len(rgb_camXs)==2)
                view1 = Image.fromarray(rgb_camXs[0])
                view2 = Image.fromarray(rgb_camXs[1])
            # print(max(view1))
            # print(min(view1))
            rgb = self.data_aug(view1, view2)
            # time2 = time.time()
            # print(3, time2-time1)
            # rgb_camXs = utils.py.preprocess_color(rgb_camXs)
            d['rgb_camXs'] = rgb
            if 'masks_camXs' in d:
                d['masks_camXs'] = list(d['masks_camXs'])
        else:
            # print("here3")
            # print(d.keys())
            if 'rgb_camXs_raw' in d:
                rgb_camXs = d['rgb_camXs_raw']
            else:
                rgb_camXs = d['rgb_camXs']
            # move channel dim inward, like pytorch wants
            rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
            rgb_camXs = utils.py.preprocess_color(rgb_camXs)
            d['rgb_camXs'] = rgb_camXs
            if 'masks_camXs' in d:
                d['masks_camXs'] = list(d['masks_camXs'])
        
        # if (self.data_format=='multiview'):
        #     # we also have camR
        #     rgb_camRs = d['rgb_camRs']
        #     rgb_camRs = np.transpose(rgb_camRs, axes=[0, 3, 1, 2])
        #     rgb_camRs = utils.py.preprocess_color(rgb_camRs)
        #     d['rgb_camRs'] = rgb_camRs
        
        d['filename'] = filename
        # time2 = time.time()
        # print(2, time2-time1)
        return d

    def __len__(self):
        return len(self.records)

    def get_item_names(self):
        if self.data_format=='seq':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_camXs',
                'rgb_camXs',
                'xyz_camXs',
                'boxlists',
                'tidlists',
                'scorelists',
            ]
        elif self.data_format=='multiview':
            # print('here')
            item_names = [
                'pix_T_cams_raw',
                'camR_T_origin_raw',
                'origin_T_camRs_raw',
                'origin_T_camXs_raw',
                'rgb_camXs_raw',
                # 'seg_camXs',
                # 'lrtlist_camRs',
                # 'rgb_camRs',
                'xyz_camXs_raw',
                # 'masks_camXs',
                # 'boxlists',
                # 'tidlist_s',
                # 'scorelist_s',
            ]
        elif self.data_format=='dino_multiview':
            item_names = [
                # 'pix_T_cams_raw',
                # 'camR_T_origin_raw',
                # 'origin_T_camRs_raw',
                # 'origin_T_camXs_raw',
                'rgb_camXs_raw',
                # 'seg_camXs',
                # 'lrtlist_camRs',
                # 'rgb_camRs',
                # 'xyz_camXs_raw',
                # 'masks_camXs',
                # 'boxlists',
                # 'tidlist_s',
                # 'scorelist_s',
            ]
        elif self.data_format=='traj':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_camXs',
                'rgb_camXs',
                'xyz_camXs',
                'box_traj_camR',
                'score_traj',
                'full_boxlist_camR',
                'full_scorelist',
                'full_tidlist',
            ]
        elif self.data_format=='complete':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_cam0s',
                'origin_T_cam1s',
                'origin_T_cam2s',
                'origin_T_cam3s',
                'rgb_camRs',
                'rgb_cam0s',
                'rgb_cam1s',
                'rgb_cam2s',
                'rgb_cam3s',
                'xyz_camRs',
                'xyz_cam0s',
                'xyz_cam1s',
                'xyz_cam2s',
                'xyz_cam3s',
                'box_traj_camR',
                'score_traj',
                'full_boxlist_camR',
                'full_scorelist',
                'full_tidlist',
            ]
        elif self.data_format=='oldtraj':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_camXs',
                'rgb_camXs',
                'xyz_camXs',
                'box_traj_camR',
                'score_traj',
            ]
        elif self.data_format=='simpletraj':
            item_names = [
                'pix_T_cams',
                'origin_T_camRs',
                'origin_T_camXs',
                'rgb_camX0',
                'xyz_camX0',
                'box_traj_camR',
                'score_traj',
            ]
        elif self.data_format=='ktrack':
            item_names = [
                'rgb_camXs',
                'xyz_veloXs',
                'origin_T_camXs',
                'pix_T_cams',
                'cams_T_velos',
                'boxlists',
                'tidlists',
                'scorelists',
            ]
        elif self.data_format=='nuscenes':
            item_names = [
                'rgb_camXs',
                'xyz_camXs',
                'origin_T_camXs',
                'pix_T_cams',
                'lrtlist_camXs',
                'scorelist_camXs',
                'tidlist_camXs',
            ]
        elif self.data_format=='kodo':
            item_names = [
                'rgb_camXs',
                'xyz_veloXs',
                'origin_T_camXs',
                'pix_T_cams',
                'cams_T_velos',
                # 'boxlists',
                # 'tidlists',
                # 'scorelists',
            ]
        else:
            item_names = None
        return item_names

    def random_select_single(self, batch, num_samples=2):
        item_names = self.get_item_names()

        # num_all = len(batch[item_names[origin]]) # total number of frames
        # print(batch.keys())
        if 'origin_T_camXs_raw' in batch.keys():
            num_all = len(batch['origin_T_camXs_raw']) # total number of frames
        else:
            num_all = len(batch['origin_T_camXs']) # total number of frames

        inds = np.random.randint(0, num_all, num_samples)

        rand_num = np.random.uniform(0,1)
        if rand_num<0.05:
            inds[1] = inds[0] # predict same image 5% of the time (like in coco setup)

        if (self.data_format=='traj') or (self.data_format=='simpletraj') or (self.data_format=='ktrack') or (self.data_format=='oldtraj') or (self.data_format=='kodo') or (self.data_format=='complete'):
            # print('loading a traj')
            if self.data_consec:
                # we want a contiguous subseq

                # print('num_all', num_all)
                # print('num_samples', num_samples)
                
                stride = 1
                # print('gathering data with stride %d' % stride)
                
                start_inds = list(range(num_all-num_samples))
                if num_all==num_samples:
                    start_ind = 0
                else:
                    start_ind = np.random.randint(0, num_all-num_samples*stride, 1).squeeze()
                # print('starting at %d' % start_ind)

                batch_new = {}
                inds = list(range(start_ind,start_ind+num_samples*stride,stride))
                for item_name in item_names:
                    if not (item_name in ['rgb_camX0', 'xyz_camX0']):
                        item = batch[item_name]
                        # item = item[start_ind:start_ind+num_samples]
                        item = item[inds]
                        batch_new[item_name] = item
                    else:
                        # copy directly
                        print(len(batch[item_name]))
                        batch_new[item_name] = batch[item_name]
            else:

                inds = np.random.randint(0, num_all, num_samples)
                # print('setting ind0 to 0')
                # inds[0] = 0
                # print('setting ind1 to 1')
                # inds[1] = 1
                
                # print('taking inds', inds)

                batch_new = {}
                for item_name in item_names:
                    if not (item_name in ['rgb_camX0', 'xyz_camX0']):
                        item = batch[item_name]
                        item = item[inds]
                        batch_new[item_name] = item
                    else:
                        # copy directly
                        print('else', len(batch[item_name]))
                        batch_new[item_name] = batch[item_name]

        elif ('object_info_s_list' in batch.keys()):
            # print(1)
            # inds = np.random.randint(0, num_all, num_samples)
            batch_new = {}
            for item_name in item_names:
                # print('item_name', item_name)
                # print('item', batch[item_name].shape)
                # if item_name not in batch.keys():
                #     continue

                if (item_name in ['object_info_s_list']):
                    item = batch[item_name]
                    item2 = [item[i] for i in list(np.array(range(len(sample_id)))[final_sample])]
                    batch_new[item_name] = item2
                # elif (item_name in ['rgb_camXs_raw']):
                #     item = batch[item_name]
                #     item = item[inds]
                #     if 'raw' in item_name:
                #         item_name = item_name.replace('_raw', '')
                #     batch_new[item_name] = item

                elif not (item_name in ['camR_index']):
                    # print(batch.keys())
                    
                    if 'raw' in item_name:
                        item_name2 = item_name.replace('_raw', '')
                    if item_name not in batch:
                        # print(item_name)
                        # print(batch.keys())
                        continue
                    item = batch[item_name]
                    # print(item.shape)
                    item = item[inds]
                    batch_new[item_name2] = item
                else:
                    # copy directly
                    batch_new[item_name] = batch[item_name]
        # elif 'masks_camXs' in batch.keys():
        #     batch_new = batch
        #     print(1)
                
        else:
            # print(2)
            # first shuffle
            # inds = np.random.permutation(list(range(num_all)))
            # inds = inds[:num_samples]

            # inds = np.random.randint(0, num_all, num_samples)
            
            # print('setting ind0 to 0')
            # inds[0] = 0
            
            batch_new = {}
            for item_name in item_names:
                # item = batch[item_name]

                if not (item_name in ['camR_index']):
                    
                    
                    if 'raw' in item_name:
                        item_name = item_name.replace('_raw', '')
                    if item_name not in batch:
                        # print(item_name)
                        # print(batch.keys())
                        continue
                        
                    item = batch[item_name]
                    item = item[inds]
                    batch_new[item_name] = item

                # print('item', item_name, len(item), item[0].shape)
                
                # item = item[inds]
                # # now select a random set of length num_samples
                # item = item[:num_samples]
                # item = item[inds]
                
                # batch_new[item_name] = item

            batch_new['ind_along_S'] = inds
        return batch_new

    def non_random_select_single(self, batch, num_samples=2):
        item_names = self.get_item_names()
        # num_all = len(batch[item_names[origin]]) # total number of frames
        # print(batch.keys())
        if 'origin_T_camXs_raw' in batch.keys():
            num_all = len(batch['origin_T_camXs_raw']) # total number of frames
        else:
            num_all = len(batch['origin_T_camXs']) # total number of frames

        if (self.data_format=='traj') or (self.data_format=='simpletraj') or (self.data_format=='ktrack') or (self.data_format=='oldtraj') or (self.data_format=='kodo') or (self.data_format=='complete'):
            # print('loading a traj')
            if self.data_consec:
                # we want a contiguous subseq

                # print('num_all', num_all)
                # print('num_samples', num_samples)
                
                stride = 1
                # print('gathering data with stride %d' % stride)
                
                start_inds = list(range(num_all-num_samples))
                if num_all==num_samples:
                    start_ind = 0
                else:
                    start_ind = np.random.randint(0, num_all-num_samples*stride, 1).squeeze()
                # print('starting at %d' % start_ind)

                batch_new = {}
                inds = list(range(start_ind,start_ind+num_samples*stride,stride))
                for item_name in item_names:
                    if not (item_name in ['rgb_camX0', 'xyz_camX0']):
                        item = batch[item_name]
                        # item = item[start_ind:start_ind+num_samples]
                        item = item[inds]
                        batch_new[item_name] = item
                    else:
                        # copy directly
                        batch_new[item_name] = batch[item_name]
            else:

                # inds = np.random.randint(0, num_all, num_samples)
                inds = np.array(list(range(num_samples)))
                # print('setting ind0 to 0')
                # inds[0] = 0
                # print('setting ind1 to 1')
                # inds[1] = 1
                
                # print('taking inds', inds)

                batch_new = {}
                for item_name in item_names:
                    if not (item_name in ['rgb_camX0', 'xyz_camX0']):
                        item = batch[item_name]
                        item = item[inds]
                        batch_new[item_name] = item
                    else:
                        # copy directly
                        batch_new[item_name] = batch[item_name]

        elif ('object_info_s_list' in batch.keys()):
            inds = np.array(list(range(num_samples))) #np.random.randint(0, num_all, num_samples)
            batch_new = {}
            for item_name in item_names:
                # print('item_name', item_name)
                # print('item', batch[item_name].shape)
                # if item_name not in batch.keys():
                #     continue

                if (item_name in ['object_info_s_list']):
                    item = batch[item_name]
                    item2 = [item[i] for i in list(np.array(range(len(sample_id)))[final_sample])]
                    batch_new[item_name] = item2
                # elif (item_name in ['rgb_camXs_raw']):
                #     item = batch[item_name]
                #     item = item[inds]
                #     if 'raw' in item_name:
                #         item_name = item_name.replace('_raw', '')
                #     batch_new[item_name] = item

                elif not (item_name in ['camR_index']):
                    item = batch[item_name]
                    item = item[inds]
                    
                    if 'raw' in item_name:
                        item_name = item_name.replace('_raw', '')
                    batch_new[item_name] = item
                else:
                    # copy directly
                    batch_new[item_name] = batch[item_name]
        elif 'masks_camXs' in batch.keys():
            batch_new = batch
                
        else:
            # first shuffle
            # inds = np.random.permutation(list(range(num_all)))
            # inds = inds[:num_samples]
            inds = np.array(list(range(num_samples)))
            
            # print('setting ind0 to 0')
            # inds[0] = 0
            
            batch_new = {}
            for item_name in item_names:
                item = batch[item_name]

                # print('item', item_name, len(item), item[0].shape)
                
                # item = item[inds]
                # # now select a random set of length num_samples
                # item = item[:num_samples]
                item = item[inds]
                
                batch_new[item_name] = item

            batch_new['ind_along_S'] = inds
        return batch_new

    def random_time_flip_single(self, sample):
        do_flip = np.random.randint(2) # 0 or 1
        item_names = self.get_item_names()
        for item_name in item_names:
            # print('flipping', item_name)
            item = sample[item_name]
            if do_flip > 0.5:
                # flip along the seq dim, which is 0
                if torch.is_tensor(item):
                    item = item.flip(0)
                else:
                    item = np.flip(item, axis=0).copy()
            sample[item_name] = item
        return sample

def get_inputs():
    dataset_filetype = hyp.dataset_filetype
    all_set_inputs = {}
    for set_name in hyp.set_names:
        if hyp.sets_to_run[set_name]:
            data_path = hyp.data_paths[set_name]
            shuffle = hyp.shuffles[set_name]
            data_format = hyp.data_formats[set_name]
            # print("DATAFORMAT", data_format)
            data_consec = hyp.data_consecs[set_name]
            seqlen = hyp.seqlens[set_name]
            batch_size = hyp.batch_sizes[set_name]
            if dataset_filetype == 'npz':
                # print('setting num_workers=4')
                # print('setting num_workers=1')
                # print(set_name, data_format)
                # num_workers = 4
                # num_workers = 8
                num_workers = 1
                print(shuffle, data_format, data_consec, seqlen, batch_size)
                # num_workers = 2
                print('setting num_workers=%d' % num_workers)
                dataset = IndexedDataset(NpzDataset(
                    dataset_path=data_path,
                    shuffle=shuffle,
                    data_format=data_format,
                    data_consec=data_consec,
                    seqlen=seqlen))
                # all_datasets[set_name] = dataset
                all_set_inputs[set_name] = torch.utils.data.DataLoader(
                    dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True)
            else:
                assert False # other filetypes not ready right now

    return all_set_inputs

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

    def __call__(self, view1, view2):
        crops = []
        crops.append(self.global_transfo1(view1))
        crops.append(self.global_transfo2(view1))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(view2))
        return crops