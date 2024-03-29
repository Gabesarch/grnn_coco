from exp_base import *

import ipdb
st = ipdb.set_trace

# the idea here is to train with momentum contrast (MOC), in a simple clean way for eccv

############## choose an experiment ##############

#current = 'builder'
# current = 'trainer'

current = '{}'.format(os.environ["exp_name"])


mod = '"{}"'.format(os.environ["run_name"]) # debug

# mod = '"test00"'
# mod = '"cr00"' # carla and replica contrastive
# mod = '"cr01"' # carla and replica contrastive

############## define experiment ##############



##################%%%%%%%  MARK Lescroart 2019 Data #############%%%%%%%%%%%
# emb only
exps['trainer_lescroart_feat3d_enc3d_occ_emb3d_vox'] = [
    'lescroart_moc', # mode
    'mujoco_offline',  # mode
    'mark_data',
    'bounds_train_replica_carla',
    'pretrained_feat3d',
    'pretrained_occ',
    '500k_iters',
    '2lr5',
    'B1',
    'train_feat3d_enc3d',
    'train_occ',
    'train_emb3d',
    'fit_vox', # with fit vox, the bounds are determined
    # 'train_view',
    # 'train_rgb',
    'log500',
    # 'vallog100',
    # 'log1',
    'snap5k', 
]

# view only
exps['trainer_lescroart_feat3d_enc3d_occ_view_vox'] = [
    'lescroart_moc', # mode
    'mujoco_offline',  # mode
    'mark_data',
    'bounds_train_replica_carla',
    # 'pretrained_feat3d',
    '400k_iters',
    '2lr5',
    'B1',
    'pretrained_feat3d',
    'pretrained_view',
    'pretrained_occ',
    'train_feat3d_enc3d',
    'train_occ',
    # 'train_emb3d',
    'train_view',
    'fit_vox',
    # 'train_rgb',
    'log500',
    # 'vallog100',
    # 'log1',
    'snap10k', 
]

# occ only
exps['trainer_lescroart_feat3d_enc3d_occ_vox'] = [
    'lescroart_moc', # mode
    'mujoco_offline',  # mode
    'mark_data',
    'bounds_train_replica_carla',
    # 'pretrained_feat3d',
    '400k_iters',
    '2lr5',
    'B1',
    # 'pretrained_feat3d',
    # 'pretrained_view',
    # 'pretrained_occ',
    'train_feat3d_enc3d',
    'train_occ',
    # 'train_emb3d',
    # 'train_view',
    'fit_vox',
    # 'train_rgb',
    'log500',
    # 'vallog100',
    # 'log1',
    'snap10k', 
]


############## net configs ##############

groups['lescroart_moc'] = ['do_lescroart_moc = True']

groups['do_midas'] = ['do_midas_depth_estimation = True']

groups['train_moc2D'] = [
    'do_moc2D = True',
    'moc2D_num_samples = 1000',
    'moc2D_coeff = 1.0',
]
groups['train_moc3D'] = [
    'do_moc3D = True',
    'moc3D_num_samples = 1000',
    'moc3D_coeff = 1.0',
]
groups['train_emb3d'] = [
    'do_emb3d = True',
    # 'emb3d_ml_coeff = 1.0',
    # 'emb3d_l2_coeff = 0.1',
    # 'emb3d_mindist = 16.0',
    # 'emb3d_num_samples = 2',
    'emb3d_mindist = 16.0',
    'emb3d_num_samples = 2',
    'emb3d_ce_coeff = 1.0',
]
# groups['train_emb3d_reduced_coeff'] = [
#     'do_emb3d = True',
#     # 'emb3d_ml_coeff = 1.0',
#     # 'emb3d_l2_coeff = 0.1',
#     # 'emb3d_mindist = 16.0',
#     # 'emb3d_num_samples = 2',
#     'emb3d_mindist = 16.0',
#     'emb3d_num_samples = 2',
#     'emb3d_ce_coeff = 0.1',
# ]
groups['train_feat2D'] = [
    'do_feat2D = True',
    'feat2D_dim = 32',
    'feat2D_smooth_coeff = 0.01',
]
######## feat3d
groups['train_feat3d_enc3d'] = [
    'do_feat3d = True',
    'feat3d_smooth_coeff = 0.0',
    'feat3d_dim = 32',
    'feat3d_arch = "enc3d"',
]
groups['train_feat3d_skip3d'] = [
    'do_feat3d = True',
    'feat3d_smooth_coeff = 0.0',
    'feat3d_dim = 32',
    'feat3d_arch = "skip3d"',
]
groups['train_feat3d_encdec3d'] = [
    'do_feat3d = True',
    'feat3d_smooth_coeff = 0.0',
    'feat3d_dim = 32',
    'feat3d_arch = "encdec3d"',
]
#########
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 1.0',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
    # 'view_l1_coeff = 0.1',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    # 'occ_smooth_coeff = 0.1',
     'occ_smooth_coeff = 1.0',
]
groups['train_rgb'] = [
    'do_rgb = True',
    'rgb_l1_coeff = 1.0',
    'rgb_smooth_coeff = 0.1',
]

groups['vallog100'] = [
    'log_freq_val = 100',
]

groups['fit_vox'] = [
    'fit_vox = True',
    # 'clip_bounds = None',
]

groups['mujoco_offline'] = ['do_mujoco_offline = True']
groups['mujoco_offline_metric'] = ['do_mujoco_offline_metric = True']
groups['mujoco_offline_metric_2d'] = ['do_mujoco_offline_metric_2d = True']
groups['touch_embed'] = ['do_touch_embed = True']



#K = 2 # how many objects to consider
#N = 8 # how many objects per npz
#S_test = 100
H = 256
W = 768
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)
# PH = int(H)
# PW = int(W)

SIZE = 36

groups['bounds_train_replica_carla'] = [
    'XMIN = -3.4', # right (neg is left)
    'XMAX = 3.4', # right
    'YMIN = -3.4', # down (neg is up)
    'YMAX = 3.4', # down
    'ZMIN = 0.0', # forward
    'ZMAX = 6.8', # forward    
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*4)),
]

# SIZE = 16
# SIZE_val = 16
# SIZE_test = 16
# SIZE_zoom = 16

# SIZE = 20
# SIZE_val = 20
# SIZE_test = 20
# SIZE_zoom = 20

# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"

# groups['carla_multiview_train_data'] = [
#     'dataset_name = "carla"',
#     'H = %d' % H,
#     'W = %d' % W,
#     'trainset = "smabs5i8t"',
#     'trainset_format = "multiview"', 
#     'trainset_seqlen = %d' % S, 
#     'dataset_location = "%s"' % dataset_location,
#     'dataset_filetype = "npz"'
# ]

groups['carla_and_replica_train'] = [
    'dataset_name = "replica"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "carla_and_replica_train"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_list_dir = "/user_data/gsarch/carla_replica_data/files"',
    'dataset_location = "/user_data/gsarch/carla_replica_data/files"',
    'dataset_filetype = "npz"',
]

groups['carla_and_replica_val'] = [
    'dataset_name = "replica"',
    'H = %d' % H,
    'W = %d' % W,
    'valset = "carla_and_replica_val"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_list_dir = "/user_data/gsarch/carla_replica_data/files"',
    'dataset_location = "/user_data/gsarch/carla_replica_data/files"',
    'dataset_filetype = "npz"',
]

groups['mark_data_small'] = ['dataset_name = "markdata"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "mark_data_small_train"',
                                 'valset = "mark_data_small_train"',
                                 'dataset_list_dir = "/lab_data/tarrlab/common/datasets/markdata"',
                                 'dataset_filetype = "txt"'
]


groups['mark_data'] = ['dataset_name = "markdata"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "mark_data_train"',
                                 'valset = "mark_data_train"',
                                 'dataset_list_dir = "/lab_data/tarrlab/common/datasets/markdata"',
                                 'dataset_location = "/lab_data/tarrlab/common/datasets/markdata"',
                                 'dataset_filetype = "txt"'
]


groups['mark_val'] = ['dataset_name = "markdata_val"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 #'trainset = "mark_testdata_test"',
                                 'valset = "mark_testdata_test"',
                                 'dataset_list_dir = "/lab_data/tarrlab/common/datasets/markdata"',
                                 #'dataset_list_dir = "/home/htung/Desktop/BlenderTemp"',
                                 'dataset_filetype = "txt"'
]

groups['mark_testdata_trn1'] = ['dataset_name = "markdata_test"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 #'trainset = "mark_testdata_test"',
                                 'testset = "mark_testdata_trn1"',
                                 'dataset_list_dir = "/lab_data/tarrlab/common/datasets/markdata"',
                                 #'dataset_list_dir = "/home/htung/Desktop/BlenderTemp"',
                                 'dataset_filetype = "txt"'
]


groups['mark_testdata_trn2'] = ['dataset_name = "markdata_test"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 #'trainset = "mark_testdata_test"',
                                 'testset = "mark_testdata_trn2"',
                                 'dataset_list_dir = "/lab_data/tarrlab/common/datasets/markdata"',
                                 #'dataset_list_dir = "/home/htung/Desktop/BlenderTemp"',
                                 'dataset_filetype = "txt"'
]

groups['mark_testdata_trn3'] = ['dataset_name = "markdata_test"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 #'trainset = "mark_testdata_test"',
                                 'testset = "mark_testdata_trn3"',
                                 #'dataset_list_dir = "/projects/katefgroup/datasets/markdata"',
                                 'dataset_list_dir = "/home/htung/Desktop/BlenderTemp"',
                                 'dataset_filetype = "txt"'
]

groups['mark_testdata_trn4'] = ['dataset_name = "markdata_test"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 #'trainset = "mark_testdata_test"',
                                 'testset = "mark_testdata_trn4"',
                                 'dataset_list_dir = "/lab_data/tarrlab/common/datasets/markdata"',
                                 #'dataset_list_dir = "/home/htung/Desktop/BlenderTemp"',
                                 'dataset_filetype = "txt"'
]

############## verify and execute ##############

def _verify_(s):
    varname, eq, val = s.split(' ')
    assert varname in globals()
    assert eq == '='
    assert type(s) is type('')

print(current)
assert current in exps
for group in exps[current]:
    print("  " + group)
    assert group in groups
    for s in groups[group]:
        print("    " + s)
        _verify_(s)
        exec(s)
        
s = "mod = " + mod
_verify_(s)

exec(s)