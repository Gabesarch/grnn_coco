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


exps['trainer_gqn_pool'] = [
    'carla_gqn', # mode
    # 'carla_surveil_multiview_train_data',
    # 'carla_multiview_train_data',
    'carla_and_replica_train',
    # '16-8-16_bounds_train',
    # '8-8-8_bounds_train',
    # 'bounds_train_replica_carla',
    # 'pretrained_feat3d',
    '200k_iters',
    'lr4',
    'B1',
    'train_gqn_pool',
    'log500',
    # 'log1',
    'snap5k', 
]

exps['val_only_gqn_pool'] = [
    'carla_gqn', # mode
    # 'carla_surveil_multiview_train_data',
    # 'carla_multiview_train_data',
    'carla_and_replica_val',
    # '16-8-16_bounds_train',
    # '8-8-8_bounds_train',
    # 'bounds_train_replica_carla',
    'pretrained_gqn',
    '1000_iters',
    'lr4',
    'B1',
    'train_gqn_pool',
    'start1',
    # 'log500',
    'no_backprop',
    'log1',
    'snap5k', 
]
# Lescroart data
exps['trainer_gqn_pool_lescroart'] = [
    'lescroart_gqn', # mode
    'mujoco_offline',  # mode
    'mark_data',
    # 'carla_surveil_multiview_train_data',
    # 'carla_multiview_train_data',
    # '16-8-16_bounds_train',
    # '8-8-8_bounds_train',
    # 'bounds_train_replica_carla',
    'pretrained_gqn',
    '200k_iters',
    'lr4',
    'B32',
    'train_gqn_pool',
    'log500',
    # 'log1',
    'snap5k', 
]

############## net configs ##############

groups['carla_gqn'] = ['do_carla_gqn = True']

groups['lescroart_gqn'] = ['do_lescroart_gqn = True']

groups['train_gqn_pool'] = [
    'do_gqn = True',
    'gqn_representation = "pool"',
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
    'trainset = "carla_and_replica_val"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
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