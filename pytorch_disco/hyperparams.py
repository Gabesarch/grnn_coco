import os
import ipdb
from munch import Munch
st = ipdb.set_trace

B = 4 # batch size
H = 240 # height
W = 320 # width

# BY = 200*2 # bird height (y axis, [-40, 40])
# BX = 176*2 # bird width (x axis, [0, 70.4])
# BZ = 20 # bird depth (z axis, [-3.0, 1.0])

# MH = 200*2
# MW = 176*2
# MD = 20

arch = "vit_small"
patch_size = 16
out_dim = 65536
norm_last_layer = True
momentum_teacher = 0.996
use_bn_in_head = False
warmup_teacher_temp = 0.04
teacher_temp = 0.04
warmup_teacher_temp_epochs = 0
use_fp16 = True
lr = 0.0005
global_crops_scale = (0.4, 1.)
local_crops_number = 8
local_crops_scale = (0.05, 0.4)
drop_path_rate = 0.1
epochs = 300
dataparallel = False
predict_view1_only = False
debug = False

assert_cube = True

Z = 128
Y = 64
X = 128

PH = int(128/4)
PW = int(384/4)

start_at_iter1 = False
do_midas_depth_estimation = False
do_lescroart_gqn = False
do_dino_multiview = False

# ZY = 32
# ZX = 32
# ZZ = 16

N = 50 # number of boxes produced by the rcnn (not all are good)
K = 1 # number of boxes to actually use
S = 2 # seq length
T = 256 # height & width of birdview map
V = 100000 # num velodyne points 

Z = 128
Y = 64
X = 128
Z_val = 128
Y_val = 64
X_val = 128
Z_test = 128
Y_test = 64
X_test = 128
Z_zoom = 128
Y_zoom = 64
X_zoom = 128

PH = int(128/4)
PW = int(384/4)

ZY = 32
ZX = 32
ZZ = 32

N = 50 # number of boxes produced by the rcnn (not all are good)
K = 1 # number of boxes to actually use
S = 2 # seq length
S_test = 3 # seq length
T = 256 # height & width of birdview map
V = 100000 # num velodyne points

do_carla_gqn = False
do_gqn = False
gqn_representation = 'pool'

# metric bounds of mem space 
XMIN = -16.0 # right (neg is left)
XMAX = 16.0 # right
YMIN = -1.0 # down (neg is up)
YMAX = 3.0 # down
ZMIN = 2.0 # forward
ZMAX = 34.0 # forward
XMIN_val = -16.0 # right (neg is left)
XMAX_val = 16.0 # right
YMIN_val = -1.0 # down (neg is up)
YMAX_val = 3.0 # down
ZMIN_val = 2.0 # forward
ZMAX_val = 34.0 # forward
XMIN_test = -16.0 # right (neg is left)
XMAX_test = 16.0 # right
YMIN_test = -1.0 # down (neg is up)
YMAX_test = 3.0 # down
ZMIN_test = 2.0 # forward
ZMAX_test = 34.0 # forward
XMIN_zoom = -16.0 # right (neg is left)
XMAX_zoom = 16.0 # right
YMIN_zoom = -1.0 # down (neg is up)
YMAX_zoom = 3.0 # down
ZMIN_zoom = 2.0 # forward
ZMAX_zoom = 34.0 # forward
FLOOR = 2.65 # ground (2.65m downward from the cam)
CEIL = (FLOOR-2.0) # 

#----------- loading -----------#
use_gt_centers = False
add_det_boxes = False
emb2D_init = ""
feat_init = ""
obj_init = ""
box_init = ""
ort_init = ""
inp_init = ""
traj_init = ""
occ_init = ""
preocc_init = ""
view_init = ""
feat3d_init = ""
gqn_init = ""
quant_init = ""
render_init = ""
vis_init = ""
flow_init = ""
ego_init = ""
total_init = ""
pixor_init = ""
det_init  = ""
reset_iter = False
use_instances_variation = False
use_instances_variation_all = False
var_coeff = 1.0
low_dict_size = False
pool_size = 1000
only_embed = False
fast_orient = False
cpu = False
do_freeze_emb2D = False
do_freeze_feat = False
do_freeze_obj = False
do_freeze_box = False
do_freeze_ort = False
do_freeze_inp = False
do_freeze_traj = False
do_freeze_occ = False
do_freeze_preocc = False
do_freeze_view = False
do_freeze_render = False
do_freeze_vis = False
do_freeze_flow = False
do_freeze_ego = False
do_resume = False
do_profile = False
do_freeze_feat3d = False
fit_vox = False
clip_bounds = None

# by default, only backprop on "train" iters
backprop_on_train = True
backprop_on_val = False
backprop_on_test = False
eval_quantize = False
# eval mode: save npys
do_eval_map = False
do_eval_recall = False # keep a buffer and eval recall within it
do_save_embs = False
do_save_ego = False
do_save_vis = False
use_det_boxes = False
summ_all = False
use_tight_bounds = False

#----------- augs -----------#
# do_aug2D = False
# do_aug3D = False
do_aug_color = False
do_time_flip = False
do_horz_flip = False
do_synth_rt = False
do_synth_nomotion = False
do_piecewise_rt = False
do_sparsify_pointcloud = 0 # choose a number here, for # pts to use

#----------- net design -----------#
# run nothing
do_emb2D = False
do_emb2D_gt = False

do_emb3D = False
do_emb3D_gt = False

do_feat = False

do_obj = False
do_box = False
do_ort = False
do_inp = False
do_traj = False

do_occ = False
do_occ_gt = False

do_preocc = False
do_view = False
do_view_gt = False

do_render = False
do_flow = False
do_ego = False
do_vis = False

do_carla_moc = False
do_omnidata_moc = False
trainset_format = "multiview"
trainset_seqlen = 2
dataset_filetype = "npz"
do_feat3d = False
feat3d_dim = 32
feat3d_smooth_coeff = 0.0
do_emb3d = False
emb3d_mindist = 16.0
emb3d_num_samples = 2
emb3d_ce_coeff = 1.0
do_rgb = False
do_save_outputs = False
#----------- emb hypers -----------#
emb2d_smooth_coeff = 0.0
emb3d_smooth_coeff = 0.0
emb2d_ml_coeff = 0.0
emb3d_ml_coeff = 0.0
emb2d_l2_coeff = 0.0
emb3d_l2_coeff = 0.0
emb2d_mindist = 0.0
emb3d_mindist = 0.0
emb2d_num_samples = 0
emb3d_num_samples = 0
emb3d_ce_coeff = 0.0

trainset_format = 'seq'
valset_format = 'seq'
testset_format = 'seq'
# should the seqdim be taken in consecutive order
trainset_consec = True
valset_consec = True
testset_consec = True

trainset_seqlen = 2
valset_seqlen = 2
testset_seqlen = 2

dataset_name = ""
seqname = ""
ind_dataset = ''

trainset = ""
valset = ""
testset = ""

dataset_location = ""

dataset_filetype = "tf" # can be tf or npz

feat3d_arch = 'enc3d'



#----------- nel utils -----------#
use_2d_boxes = False
vq_rotate = False
moc = False
moc_qsize = 1000
emb3D_o = False
eval_recall_o=False
debug_eval_recall_o=False
do_orientation = False
eval_recall_summ_o=False
do_debug = False
break_constraint = False
builder_big = False
profile_time = False
low_res = False
random_noise = False
gt = False
do_emb3D_moc = False
no_bn = True
imgnet = False
sudo_backprop = True
only_cs_vis = False
self_improve_once = False
self_improve_iterate = False
cs_filter = False
only_q_vis =  False
exp_log_freq = 100
replace_with_cs = False
maxm_log_freq = 100
exp_max_iters = 1000
maxm_max_iters = 100
self_improve_iterate = False
det_pool_size = 1000

do_metric_learning = False
do_mujoco_offline = False
do_mujoco_offline_metric = False
do_mujoco_offline_metric_2d = False
do_touch_embed = False

exp_do =False
max_do = False
exp_done = False

alpha_pos = 1.5
beta_neg = 1.0
high_neg = False
maskout = False

max_samples = None

imgnet_v1 = True
eval_recall_log_freq = 5
# make sure it is a multiple of eval_recall_log_freq
# showld be smaller than step for 
eval_compute_freq = 99
deeper_det = False
do_det = False
offline_cluster = False
offline_cluster_pool_size = 100000
cluster_vox = False
hard_vis = False
online_cluster = False
object_quantize_dictsize = 41
object_quantize_init = None
detach_background = True
quantize_loss_coef = 1.0
create_example_dict = False
filter_boxes = False
dict_distance_thresh = 1500.0
neg_cs_thresh = 0.6
pos_cs_thresh = 0.5

dataset_name1 = None
trainset1 = None
valset1 = None
dataset_list_dir1 = None
dataset_location1 = None
dataset_filetype1 = None
trainset_format1 = None
trainset_seqlen1 = None
dataset_name2 = None
trainset2 = None
valset2 = None
dataset_list_dir2 = None
dataset_location2 = None
dataset_filetype2 = None
trainset_format2 = None
trainset_seqlen2 = None
dataset_name3 = None
trainset3 = None
valset3 = None
dataset_list_dir3 = None
dataset_location3 = None
dataset_filetype3 = None
trainset_format3 = None
trainset_seqlen3 = None
valset_format1 = None
valset_seqlen1 = None

do_pixor_det = False
do_gt_pixor_det = False
pixor_alpha = 1.0
pixor_beta = 0.1
calculate_mean = False
calculate_std = False
use_pixor_focal_loss = False

online_cluster_eval = False
cluster_iters = 10000
initial_cluster_size = 10000
gt_rotate_combinations = False
obj_multiview = False
object_ema = False
object_quantize = False
object_quantize_dictsize = 41
object_quantize_comm_cost = 0.25
object_quantize_init = None

voxel_quantize = False
voxel_quantize_dictsize = 512
voxel_quantize_comm_cost = 0.25
voxel_quantize_init = None

use_kmeans = False
use_vq_vae = False

offline_cluster_eval_iters = 1000
offline_cluster_eval = False
num_rand_samps = 10
#----------- parent groups -----------#

for custom_exp in ["exp"]:
    exec(f"{custom_exp} = Munch()")
    exec(f"{custom_exp}.do = False")
    exec(f"{custom_exp}.do_debug = False")    
    exec(f"{custom_exp}.tdata = False")
    exec(f"{custom_exp}.log_freq =100")
    exec(f"{custom_exp}.max_iters = 10")
    exec(f"{custom_exp}.no_update = False")



for custom_max in ["max"]:
    exec(f"{custom_max} = Munch()")
    exec(f"{custom_max}.do = False")
    exec(f"{custom_max}.tdata = False")
    exec(f"{custom_max}.B = 1")
    exec(f"{custom_max}.max_iters = 10")
    exec(f"{custom_max}.log_freq =100")
    exec(f"{custom_max}.shuffle = False")
    exec(f"{custom_max}.g_max_iters = 0")
    exec(f"{custom_max}.p_max_iters = 1")
    exec(f"{custom_max}.predicted_matching = False")
    exec(f"{custom_max}.num_patches_per_emb = 10")
    exec(f"{custom_max}.hardmining = False")
    exec(f"{custom_max}.tripleLossThreshold = 0.8")
    exec(f"{custom_max}.max_epochs = 1")
    exec(f"{custom_max}.hardmining_gt = False")    
    exec(f"{custom_max}.searchRegion = 2") #size of the candidate proposal region
    exec(f"{custom_max}.shouldResizeToRandomScale = False") #should scale randomly or not
    exec(f"{custom_max}.margin = 3") #margin  while selecting the candidate proposals (select everything within the margin) (should be half of valid Region)!
    exec(f"{custom_max}.numRetrievalsForEachQueryEmb = 20") # the number of retrieval in the pool to consider while finding candidate prooposal region
    exec(f"{custom_max}.topK = 10") #the number of top candidate proposals to use based on simple matching (default value used in paper is 10 but in our case it should be less than numRetrievalsForEachQueryEmb or it will give a bug!)
    exec(f"{custom_max}.nbImgEpoch = 200") #the number of top pos pairs to use based on selection verification (default is 200)
    exec(f"{custom_max}.trainRegion = 6") #the regions distance from the center matches to train on (works better if a be  bit bigger than valid Region as mentioned in the paper) 
    exec(f"{custom_max}.validRegion = 6") #region to consider while trying to validate retreival (should be double of margin!)
    exec(f"{custom_max}.visualizeHardMines = False") #region to consider while trying to validate retreival (should be double of margin!)
    exec(f"{custom_max}.hard_moc = False")
    exec(f"{custom_max}.hard_moc_qsize = 100")
    exec(f"{custom_max}.exceptions = False")




for custom_max in ["emb_moc"]:
    exec(f"{custom_max} = Munch()")
    exec(f"{custom_max}.do = False")
    exec(f"{custom_max}.max_iters_init = 1000")
    exec(f"{custom_max}.max_pool_indices = 1000")
    exec(f"{custom_max}.indexes_to_take = 128")
    exec(f"{custom_max}.normal_queue = True")
    exec(f"{custom_max}.own_data_loader = False")


#----------- general hypers -----------#
lr = 0.0
delete_old_checkpoints = True
delete_checkpoints_older_than = 3


#----------- emb hypers -----------#
emb_2D_smooth_coeff = 0.0
emb_3D_smooth_coeff = 0.0
emb_2D_ml_coeff = 0.0
emb_3D_ml_coeff = 0.0
emb_2D_l2_coeff = 0.0
emb_3D_l2_coeff = 0.0
emb_2D_mindist = 0.0
emb_3D_mindist = 0.0
emb_2D_num_samples = 0
emb_3D_num_samples = 0
do_object_specific = False
hard_eval = False
use_first_bbox = True
#  have3n't still return the code for nel to handle multiple boxes!!!!!
moc_2d = False
#----------- feat hypers -----------#
feat_coeff = 0.0
feat_rigid_coeff = 0.0
feat_do_vae = False
feat_do_sb = False
feat_do_resnet = False
feat_do_sparse_invar = False
feat_quantize = False
feat_quantize_dictsize = 32
feat_quantize_comm_cost = 0.25
feat_quantize_init = None
feat_kl_coeff = 0.0
feat_dim = 8
feat_do_flip = False
feat_do_rt = False

#----------- obj hypers -----------#
obj_coeff = 0.0
obj_dim = 8

#----------- box hypers -----------#
box_sup_coeff = 0.0
box_cs_coeff = 0.0
box_dim = 8

#----------- ort hypers -----------#
ort_coeff = 0.0
ort_warp_coeff = 0.0
ort_dim = 8

#----------- inp hypers -----------#
inp_coeff = 0.0
inp_dim = 8

#----------- traj hypers -----------#
traj_coeff = 0.0
traj_dim = 8

#----------- preocc hypers -----------#
preocc_do_flip = False
preocc_coeff = 0.0
preocc_smooth_coeff = 0.0
preocc_reg_coeff = 0.0
preocc_density_coeff = 0.0

#----------- occ hypers -----------#
occ_do_cheap = False
occ_coeff = 0.0
occ_smooth_coeff = 0.0

#----------- view hypers -----------#
view_depth = 64
view_accu_render = False
view_accu_render_unps = False
view_accu_render_gt = False
view_pred_embs = False
view_pred_rgb = False
view_l1_coeff = 0.0
view_ce_coeff = 0.0
view_dl_coeff = 0.0

#----------- render hypers -----------#
render_depth = 64
render_embs = False
render_rgb = False
render_l1_coeff = 0.0

#----------- vis hypers-------------#
vis_softmax_coeff = 0.0
vis_hard_coeff = 0.0
vis_l1_coeff = 0.0
vis_debug = False

#----------- flow hypers -----------#
flow_warp_coeff = 0.0
flow_warp_g_coeff = 0.0
flow_cycle_coeff = 0.0
flow_smooth_coeff = 0.0
flow_hinge_coeff = 0.0
flow_l1_coeff = 0.0
flow_l2_coeff = 0.0
flow_synth_l1_coeff = 0.0
flow_synth_l2_coeff = 0.0
flow_do_synth_rt = False
flow_heatmap_size = 4

#----------- ego hypers -----------#
ego_use_gt = False
ego_use_precomputed = False
ego_rtd_coeff = 0.0
ego_rta_coeff = 0.0
ego_traj_coeff = 0.0
ego_warp_coeff = 0.0

#----------- mod -----------#

mod = '""'

#----------- nel params -----------#

do_empty = False
do_eval_boxes = False

############ slower-to-change hyperparams below here ############
root_keyword ="katefgroup"
demo_file_save_root_location = "/home/shamitl/datasets/demo_veggies"
det_anchor_size = 12.0
det_prob_coeff = 1.0
det_reg_coeff = 1.0

## logging
randomly_select_views = False
log_freq_train = 100
log_freq_val = 100
vis_clusters = False
log_freq_test = 100
log_freq = 100
snap_freq = 5000

max_iters = 10000
shuffle_train = True
shuffle_val = True
shuffle_test = True

dataset_name = ""
seqname = ""

trainset = ""
valset = ""
testset = ""

dataset_list_dir = ""
dataset_location = ""
root_dataset = ""

dataset_format = "tf" #can be tf or npz

# mode selection
use_bounds=False
do_zoom = False
do_carla_det = False
do_carla_mot = False
do_carla_sta = False
do_clevr_sta = False
do_nel_sta = False
do_carla_flo = False
do_carla_obj = False
do_mujoco_offline = False
do_lescroart_moc = False

GENERATE_PCD_BBOX = False

identifier_self_define = ""
############ rev up the experiment ############
mode = os.environ["MODE"]
print('os.environ mode is %s' % mode)
if mode=="CARLA_DET":
    exec(compile(open('exp_carla_det.py').read(), 'exp_carla_det.py', 'exec'))
elif mode=="CARLA_MOT":
    exec(compile(open('exp_carla_mot.py').read(), 'exp_carla_mot.py', 'exec'))
elif mode=="CARLA_FLO":
    exec(compile(open('exp_carla_flo.py').read(), 'exp_carla_flo.py', 'exec'))
elif mode=="CARLA_MOC":
    exec(compile(open('exp_carla_moc.py').read(), 'exp_carla_moc.py', 'exec'))
elif mode=="CARLA_GQN":
    exec(compile(open('exp_carla_gqn.py').read(), 'exp_carla_gqn.py', 'exec'))
elif mode=="CARLA_OBJ":
    exec(compile(open('exp_carla_obj.py').read(), 'exp_carla_obj.py', 'exec'))
elif mode=="CARLA_STA":
    exec(compile(open('exp_carla_sta.py').read(), 'exp_carla_sta.py', 'exec'))
elif mode=="NEL_STA":
    exec(compile(open('exp_nel_sta.py').read(), 'exp_nel_sta.py', 'exec'))
elif mode=="DINO_MULTIVIEW":
    exec(compile(open('exp_dino_multiview.py').read(), 'exp_dino_multiview.py', 'exec'))
elif mode=="CLEVR_STA":
    exec(compile(open('exp_clevr_sta.py').read(), 'exp_clevr_sta.py', 'exec'))
elif mode=="MUJOCO_OFFLINE":
    exec(compile(open('exp_mujoco_offline.py').read(), 'exp_mujoco_offline.py', 'exec'))
elif mode=="CUSTOM":
    exec(compile(open('exp_custom.py').read(), 'exp_custom.py', 'exec'))
elif mode=="LESCROART_MOC":
    exec(compile(open('exp_lescroart_moc.py').read(), 'exp_lescroart_moc.py', 'exec'))
elif mode=="LESCROART_GQN":
    exec(compile(open('exp_carla_gqn.py').read(), 'exp_carla_gqn.py', 'exec'))
elif mode=="OMNIDATA_MOC":
    exec(compile(open('exp_omnidata_moc.py').read(), 'exp_omnidata_moc.py', 'exec'))
else:
    assert(False) # what mode is this?

############ make some final adjustments ############
# if not do_mujoco_offline:
#     trainset_path = "%s/%s.txt" % (dataset_list_dir, trainset)
#     valset_path = "%s/%s.txt" % (dataset_list_dir, valset)
#     testset_path = "%s/%s.txt" % (dataset_list_dir, testset)
# else:
#     trainset_path = "%s/%s.npy" % (dataset_location, trainset)
#     valset_path = "%s/%s.npy" % (dataset_location, valset)
#     testset_path = "%s/%s.npy" % (dataset_location, testset)

trainset_path = "%s/%s.txt" % (dataset_list_dir, trainset)
valset_path = "%s/%s.txt" % (dataset_list_dir, valset)
testset_path = "%s/%s.txt" % (dataset_list_dir, testset)

trainset_batch_size = B
valset_batch_size = B
testset_batch_size = B

data_paths = {}
data_paths['train'] = trainset_path
data_paths['val'] = valset_path
data_paths['test'] = testset_path

set_nums = {}
set_nums['train'] = 0
set_nums['val'] = 1
set_nums['test'] = 2

set_names = ['train', 'val']

log_freqs = {}
log_freqs['train'] = log_freq_train
log_freqs['val'] = log_freq_val
log_freqs['test'] = log_freq_test

shuffles = {}
shuffles['train'] = shuffle_train
shuffles['val'] = shuffle_val
shuffles['test'] = shuffle_test

data_formats = {}
data_formats['train'] = trainset_format
data_formats['val'] = valset_format
data_formats['test'] = testset_format

data_consecs = {}
data_consecs['train'] = trainset_consec
data_consecs['val'] = valset_consec
data_consecs['test'] = testset_consec

seqlens = {}
seqlens['train'] = trainset_seqlen
seqlens['val'] = valset_seqlen
seqlens['test'] = testset_seqlen

batch_sizes = {}
batch_sizes['train'] = trainset_batch_size
batch_sizes['val'] = valset_batch_size
batch_sizes['test'] = testset_batch_size


############ autogen a name; don't touch any hypers! ############

def strnum(x):
    s = '%g' % x
    if '.' in s:
        s = s[s.index('.'):]
    return s

name = "%02d_m%dx%dx%d" % (B, Z,Y,X)
if do_view or do_emb2D or do_render:
    name += "_p%dx%d" % (PH,PW)

if lr > 0.0:
    lrn = "%.1e" % lr
    # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1]
    name += "_%s" % lrn

if do_preocc:
    name += "_P"
    if preocc_do_flip:
        name += "l"
    if do_freeze_preocc:
        name += "f"
    preocc_coeffs = [
        preocc_coeff,
        preocc_smooth_coeff,
        preocc_reg_coeff,
        preocc_density_coeff,
    ]
    preocc_prefixes = [
        "c",
        "s",
        "r",
        "d",
    ]
    for l_, l in enumerate(preocc_coeffs):
        if l > 0:
            name += "_%s%s" % (preocc_prefixes[l_],strnum(l))
            
if do_feat:
    name += "_F"
    name += "%d" % feat_dim
    if feat_do_flip:
        name += "l"
    if feat_do_rt:
        name += "r"
    if feat_do_vae:
        name += "v"
    if feat_do_sb:
        name += 'b'
    if feat_do_resnet:
        name += 'r'
    if feat_do_sparse_invar:
        name += 'i'
    if do_freeze_feat:
        name += "f"
    if feat_quantize:
        name += "q"
    else:
        feat_losses = [feat_rigid_coeff,
                       feat_kl_coeff,
        ]
        feat_prefixes = ["r",
                         "k",
        ]
        for l_, l in enumerate(feat_losses):
            if l > 0:
                name += "_%s%s" % (feat_prefixes[l_],strnum(l))


if do_ego:
    name += "_G"
    if ego_use_gt:
        name += "gt"
    elif ego_use_precomputed:
        name += "pr"
    else:
        if do_freeze_ego:
            name += "f"
        else:
            ego_losses = [ego_rtd_coeff,
                          ego_rta_coeff,
                          ego_traj_coeff,
                          ego_warp_coeff,
            ]
            ego_prefixes = ["rtd",
                            "rta",
                            "t",
                            "w",
            ]
            for l_, l in enumerate(ego_losses):
                if l > 0:
                    name += "_%s%s" % (ego_prefixes[l_],strnum(l))

if do_obj:
    name += "_J"
    # name += "%d" % obj_dim

    if do_freeze_obj:
        name += "f"
    else:
        # no real hyps here
        pass

if do_box:
    name += "_B"
    # name += "%d" % box_dim

    if do_freeze_box:
        name += "f"
    else:
        box_coeffs = [box_sup_coeff,
                      box_cs_coeff,
                      # box_smooth_coeff,
        ]
        box_prefixes = ["su",
                        "cs",
                        # "s",
        ]
        for l_, l in enumerate(box_coeffs):
            if l > 0:
                name += "_%s%s" % (box_prefixes[l_],strnum(l))


if do_ort:
    name += "_O"
    # name += "%d" % ort_dim

    if do_freeze_ort:
        name += "f"
    else:
        ort_coeffs = [ort_coeff,
                      ort_warp_coeff,
                      # ort_smooth_coeff,
        ]
        ort_prefixes = ["c",
                        "w",
                        # "s",
        ]
        for l_, l in enumerate(ort_coeffs):
            if l > 0:
                name += "_%s%s" % (ort_prefixes[l_],strnum(l))

if do_inp:
    name += "_I"
    # name += "%d" % inp_dim

    if do_freeze_inp:
        name += "f"
    else:
        inp_coeffs = [inp_coeff,
                      # inp_smooth_coeff,
        ]
        inp_prefixes = ["c",
                        # "s",
        ]
        for l_, l in enumerate(inp_coeffs):
            if l > 0:
                name += "_%s%s" % (inp_prefixes[l_],strnum(l))

if do_traj:
    name += "_T"
    name += "%d" % traj_dim

    if do_freeze_traj:
        name += "f"
    else:
        # no real hyps here
        pass

if do_occ:
    name += "_O"
    if occ_do_cheap:
        name += "c"
    if do_freeze_occ:
        name += "f"
    occ_coeffs = [occ_coeff,
                  occ_smooth_coeff,
    ]
    occ_prefixes = ["c",
                    "s",
    ]
    for l_, l in enumerate(occ_coeffs):
        if l > 0:
            name += "_%s%s" % (occ_prefixes[l_],strnum(l))

if do_view:
    name += "_V"
    if view_pred_embs:
        name += "e"
    if view_pred_rgb:
        name += "r"
    if view_accu_render:
        name += 'a'
    if view_accu_render_unps:
        name += 'u'
    if view_accu_render_gt:
        name += 'g'
    if do_freeze_view:
        name += "f"

    # sometimes, even if view is frozen, we use the loss
    # to train other nets
    view_coeffs = [view_depth,
                   view_l1_coeff,
                   view_ce_coeff,
                   view_dl_coeff,
    ]
    view_prefixes = ["d",
                     "c",
                     "e",
                     "s",
    ]
    for l_, l in enumerate(view_coeffs):
        if l > 0:
            name += "_%s%s" % (view_prefixes[l_],strnum(l))

if do_render:
    name += "_R"
    if render_embs:
        name += "e"
    if render_rgb:
        name += "r"
    if do_freeze_render:
        name += "f"

    render_coeffs = [
        render_depth,
        render_l1_coeff,
    ]
    render_prefixes = [
        "d",
        "c",
    ]
    for l_, l in enumerate(render_coeffs):
        if l > 0:
            name += "_%s%s" % (render_prefixes[l_],strnum(l))

if do_vis:
    name += "_V"
    if vis_debug:
        name += 'd'
    if do_freeze_vis:
        name += "f"
    else:
        vis_coeffs = [vis_softmax_coeff,
                      vis_hard_coeff,
                      vis_l1_coeff,
        ]
        vis_prefixes = ["s",
                        "h",
                        "c",
        ]
        for l_, l in enumerate(vis_coeffs):
            if l > 0:
                name += "_%s%s" % (vis_prefixes[l_],strnum(l))


if do_emb2D:
    name += "_E2"
    if do_freeze_emb2D:
        name += "f"
    emb_coeffs = [emb_2D_smooth_coeff,
                  emb_2D_ml_coeff,
                  emb_2D_l2_coeff,
                  emb_2D_num_samples,
                  emb_2D_mindist,
    ]
    emb_prefixes = ["s",
                    "m",
                    "e",
                    "n",
                    "d",
    ]
    for l_, l in enumerate(emb_coeffs):
        if l > 0:
            name += "_%s%s" % (emb_prefixes[l_],strnum(l))
if do_emb3D:
    name += "_E3"
    emb_coeffs = [emb_3D_smooth_coeff,
                  emb_3D_ml_coeff,
                  emb_3D_l2_coeff,
                  emb_3D_num_samples,
                  emb_3D_mindist,
    ]
    emb_prefixes = ["s",
                    "m",
                    "e",
                    "n",
                    "d",
    ]
    for l_, l in enumerate(emb_coeffs):
        if l > 0:
            name += "_%s%s" % (emb_prefixes[l_],strnum(l))

if do_flow:
    name += "_F"
    if do_freeze_flow:
        name += "f"
    else:
        flow_coeffs = [flow_heatmap_size,
                       flow_warp_coeff,
                       flow_warp_g_coeff,
                       flow_cycle_coeff,
                       flow_smooth_coeff,
                       flow_hinge_coeff,
                       flow_l1_coeff,
                       flow_l2_coeff,
                       flow_synth_l1_coeff,
                       flow_synth_l2_coeff,
        ]
        flow_prefixes = ["h",
                         "w",
                         "g",
                         "c",
                         "s",
                         "h",
                         "e",
                         "f",
                         "y",
                         "x",
        ]
        for l_, l in enumerate(flow_coeffs):
            if l > 0:
                name += "_%s%s" % (flow_prefixes[l_],strnum(l))

##### end model description

# add some training data info

sets_to_run = {}
if trainset:
    name = "%s_%s" % (name, trainset)
    sets_to_run['train'] = True
else:
    sets_to_run['train'] = False

if valset:
    name = "%s_%s" % (name, valset)
    sets_to_run['val'] = True
else:
    sets_to_run['val'] = False

if testset:
    name = "%s_%s" % (name, testset)
    sets_to_run['test'] = True
else:
    sets_to_run['test'] = False

sets_to_backprop = {}
sets_to_backprop['train'] = backprop_on_train
sets_to_backprop['val'] = backprop_on_val
sets_to_backprop['test'] = backprop_on_test


if (do_aug_color or
    do_horz_flip or
    do_time_flip or
    do_synth_rt or
    do_piecewise_rt or
    do_synth_nomotion or
    do_sparsify_pointcloud):
    name += "_A"
    if do_aug_color:
        name += "c"
    if do_horz_flip:
        name += "h"
    if do_time_flip:
        name += "t"
    if do_synth_rt:
        assert(not do_piecewise_rt)
        name += "s"
    if do_piecewise_rt:
        assert(not do_synth_rt)
        name += "p"
    if do_synth_nomotion:
        name += "n"
    if do_sparsify_pointcloud:
        name += "v"

# if (not shuffle_train) or (not shuffle_val) or (not shuffle_test):
name += "_ns"


if do_profile:
    name += "_PR"

if mod:
    name = "%s_%s" % (name, mod)
if len(identifier_self_define) > 0:
    name += ('_' + identifier_self_define)

if do_resume:
    total_init = name
    name += '_gt'

# st()
print(name)
