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

exps['trainer_carla_rep_lesc_dino_mult'] = [
    'lescroart_moc', # mode
    'mujoco_offline',  # mode
    # 'mark_data',
    'carla_replica_mark_data',
    'bounds_train_replica_carla',
    # 'pretrained_feat3d',
    # 'pretrained_occ',
    # 'resume_train',
    'dino_default_params',
    # '500k_iters',
    '45k_iters',
    # '2lr5',
    # 'B3',
    # 'train_feat3d_enc3d',
    # 'train_occ',
    # 'train_emb3d',
    # 'fit_vox', # with fit vox, the bounds are determined
    # 'train_view',
    # 'train_rgb',
    # 'log100',
    # 'log500',
    # 'vallog100',
    # 'vallog1',
    'log1',
    'snap5k', 
]

exps['trainer_carla_rep_dino_mult'] = [
    'lescroart_moc', # mode
    'mujoco_offline',  # mode
    # 'mark_data',
    'carla_replica_data',
    'bounds_train_replica_carla',
    # 'pretrained_feat3d',
    # 'pretrained_occ',
    # 'resume_train',
    'dino_default_params',
    '500k_iters',
    # '45k_iters',
    # '2lr5',
    # 'B3',
    # 'train_feat3d_enc3d',
    # 'train_occ',
    # 'train_emb3d',
    # 'fit_vox', # with fit vox, the bounds are determined
    # 'train_view',
    # 'train_rgb',
    # 'log100',
    'log500',
    # 'vallog100',
    # 'vallog1',
    # 'log1',
    'snap5k', 
]

exps['trainer_carla_rep_dino_mult_sameview'] = [
    'lescroart_moc', # mode
    'mujoco_offline',  # mode
    # 'mark_data',
    'carla_replica_data',
    'bounds_train_replica_carla',
    # 'pretrained_feat3d',
    # 'pretrained_occ',
    # 'resume_train',
    'dino_default_params',
    'predict_same_view',
    '500k_iters',
    # '45k_iters',
    # '2lr5',
    # 'B3',
    # 'train_feat3d_enc3d',
    # 'train_occ',
    # 'train_emb3d',
    # 'fit_vox', # with fit vox, the bounds are determined
    # 'train_view',
    # 'train_rgb',
    # 'log100',
    'log500',
    # 'vallog100',
    # 'vallog1',
    # 'log1',
    'snap5k', 
]

############## net configs ##############

groups['carla_moc'] = ['do_carla_moc = True']

groups['do_midas'] = ['do_midas_depth_estimation = True']

# groups['lescroart_moc'] = [
#     'do_lescroart_moc = True',
#     'do_carla_moc = True'
#     ]
groups['lescroart_moc'] = [
    'do_dino_multiview = True',
    ]
groups['mujoco_offline'] = ['do_mujoco_offline = True']
groups['mujoco_offline_metric'] = ['do_mujoco_offline_metric = True']
groups['mujoco_offline_metric_2d'] = ['do_mujoco_offline_metric_2d = True']
groups['touch_embed'] = ['do_touch_embed = True']

groups['resume_train'] = [
    'total_init = "03_m144x144x144_2e-5_O_c1_s.1_ns_carl_rep_lesc02"',
]

groups['predict_same_view'] = [
    'predict_view1_only = True',
    'S = 1',
]

groups['dino_default_params'] = [
    'arch = "vit_small"',
    'patch_size = 16',
    'out_dim = 65536',
    'norm_last_layer = True',
    'momentum_teacher = 0.996',
    'use_bn_in_head = False',
    'warmup_teacher_temp = 0.04',
    'teacher_temp = 0.04',
    'warmup_teacher_temp_epochs = 0',
    'use_fp16 = True',
    'lr = 0.00001',
    # 'lr = 0.0001',
    'global_crops_scale = (0.4,1.)',
    'local_crops_number = 8',
    'local_crops_scale = (0.05,0.4)',
    'drop_path_rate = 0.1',
    'B = 64',
]

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


groups['carla_replica_mark_data'] = [ #'dataset_name = "markdata"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'dataset_name1 = "markdata"',
                                 'trainset1 = "mark_data_train"',
                                #  'trainset_format1 = "multiview"', 
                                 'trainset_format1 = "dino_multiview"', 
                                 'trainset_seqlen1 = %d' % S, 
                                 'valset1 = "mark_data_val"',
                                 'dataset_list_dir1 = "/lab_data/tarrlab/common/datasets/markdata"',
                                 'dataset_location1 = "/lab_data/tarrlab/common/datasets/markdata"',
                                 'dataset_filetype1 = "txt"',
                                 'dataset_name2 = "replica"',
                                 'trainset2 = "train"',
                                #  'trainset_format2 = "multiview"', 
                                 'trainset_format2 = "dino_multiview"',
                                 'trainset_seqlen2 = %d' % S, 
                                #  'valset2 = "mark_data_train"',
                                 'dataset_list_dir2 = "/user_data/gsarch/carla_replica_data/files/replica_all_obj_processed/npy"',
                                 'dataset_location2 = "/user_data/gsarch/carla_replica_data/files/replica_all_obj_processed/npy"',
                                 'dataset_filetype2 = "npz"',
                                 'dataset_name3 = "replica"',
                                 'trainset3 = "viewseg_multiview_mr07_s29_i1_train"',
                                #  'trainset_format3 = "multiview"', 
                                 'trainset_format3 = "dino_multiview"',
                                 'trainset_seqlen3 = %d' % S, 
                                #  'valset3 = "mark_data_train"',
                                 'dataset_list_dir3 = "/user_data/gsarch/carla_replica_data/datasets/carla/processed/npzs"',
                                 'dataset_location3 = "/user_data/gsarch/carla_replica_data/datasets/carla/processed/npzs"',
                                 'dataset_filetype3 = "npz"',
]

groups['carla_replica_data'] = [ #'dataset_name = "markdata"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                #  'dataset_name1 = "markdata"',
                                #  'trainset1 = "mark_data_train"',
                                # #  'trainset_format1 = "multiview"', 
                                #  'trainset_format1 = "dino_multiview"', 
                                #  'trainset_seqlen1 = %d' % S, 
                                 'valset1 = "val"',
                                 'valset_format1 = "dino_multiview"',
                                 'valset_seqlen1 = %d' % S, 
                                #  'dataset_list_dir1 = "/lab_data/tarrlab/common/datasets/markdata"',
                                #  'dataset_location1 = "/lab_data/tarrlab/common/datasets/markdata"',
                                #  'dataset_filetype1 = "txt"',
                                 'dataset_name1 = "replica"',
                                 'trainset1 = "train"',
                                #  'trainset_format2 = "multiview"', 
                                 'trainset_format1 = "dino_multiview"',
                                 'trainset_seqlen1 = %d' % S, 
                                #  'valset2 = "mark_data_train"',
                                 'dataset_list_dir1 = "/user_data/gsarch/carla_replica_data/files/replica_all_obj_processed/npy"',
                                 'dataset_location1 = "/user_data/gsarch/carla_replica_data/files/replica_all_obj_processed/npy"',
                                 'dataset_filetype1 = "npz"',
                                 'dataset_name2 = "replica"',
                                 'trainset2 = "viewseg_multiview_mr07_s29_i1_train"',
                                #  'trainset_format3 = "multiview"', 
                                 'trainset_format2 = "dino_multiview"',
                                 'trainset_seqlen2 = %d' % S, 
                                #  'valset3 = "mark_data_train"',
                                 'dataset_list_dir2 = "/user_data/gsarch/carla_replica_data/datasets/carla/processed/npzs"',
                                 'dataset_location2 = "/user_data/gsarch/carla_replica_data/datasets/carla/processed/npzs"',
                                 'dataset_filetype2 = "npz"',
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