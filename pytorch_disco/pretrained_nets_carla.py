feat_init = ""
view_init = ""
flow_init = ""
emb2D_init = ""
vis_init = ""
occ_init = ""
ego_init = ""
tow_init = ""
preocc_init = ""
quant_init = ""
pixor_init = ""
gqn_init = ""
feat3d_init = ""
emb3d_init = ""

# emb_dim = 8
# # occ_cheap = False
# feat_dim = 32
# feat_do_vae = False

# view_depth = 32
# view_pred_rgb = True
# view_use_halftanh = True
# view_pred_embs = False

# occ_do_cheap = False
# this is the previous winner net, from which i was able to train a great flownet in 500i
# feat3d_init = "01_m144x144x144_p128x384_1e-4_O_c1_s.1_V_d32_c1_carla_and_replica_train_ns_enc3d_view07_fitvox"
# view_init = "01_m144x144x144_p128x384_1e-4_O_c1_s.1_V_d32_c1_carla_and_replica_train_ns_enc3d_view07_fitvox"
# occ_init = "01_m144x144x144_p128x384_1e-4_O_c1_s.1_V_d32_c1_carla_and_replica_train_ns_enc3d_view07_fitvox"

feat3d_init = "01_m144x144x144_p128x384_1e-4_O_c1_s.1_V_d32_c7_carla_and_replica_train_carla_and_replica_val_ns_enc3d_view01_fitvox_withval"
view_init = "01_m144x144x144_p128x384_1e-4_O_c1_s.1_V_d32_c7_carla_and_replica_train_carla_and_replica_val_ns_enc3d_view01_fitvox_withval"
occ_init = "01_m144x144x144_p128x384_1e-4_O_c1_s.1_V_d32_c7_carla_and_replica_train_carla_and_replica_val_ns_enc3d_view01_fitvox_withval"

# gqn_init = "01_m128x64x128_1e-4_carla_and_replica_train_ns_gqn_pool_rel00"
# feat3d_init = "01_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_ns_enc3d_emb3d_04_fitvox"
# occ_init = "01_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_ns_enc3d_emb3d_04_fitvox"
# emb3d_init = "01_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_ns_enc3d_emb3d_04_fitvox"

# feat3d_init = "01_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_ns_enc3d_emb3d_06_fitvox"
# emb3d_init = "01_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_ns_enc3d_emb3d_06_fitvox"
# occ_init = "01_m144x144x144_1e-4_O_c1_s.1_carla_and_replica_train_ns_enc3d_emb3d_06_fitvox"

# emb3d
# feat3d_init = "01_m144x144x144_1e-4_O_c1_s1_mark_data_train_mark_data_train_ns_enc3d_emb3d_00_fitvox"
# view_init = ""
# occ_init = "01_m144x144x144_1e-4_O_c1_s1_mark_data_train_mark_data_train_ns_enc3d_emb3d_00_fitvox"
feat3d_init = "01_m144x144x144_2e-5_O_c1_s1_mark_data_train_mark_data_train_ns_enc3d_emb3d_01_fitvox"
view_init = ""
occ_init = "01_m144x144x144_2e-5_O_c1_s1_mark_data_train_mark_data_train_ns_enc3d_emb3d_01_fitvox"

feat3d_init = "02_m144x144x144_1e-4_O_c1_carla_and_replica_train_carla_and_replica_val_ns_fixfeb7_16"
occ_init = "02_m144x144x144_1e-4_O_c1_carla_and_replica_train_carla_and_replica_val_ns_fixfeb7_16"

feat3d_init = "04_m144x36x144_1e-4_O_c1_s1_carla_and_replica_train_carla_and_replica_val_ns_omnidata_moc_boundsadjusted_occonly_00"
occ_init = "04_m144x36x144_1e-4_O_c1_s1_carla_and_replica_train_carla_and_replica_val_ns_omnidata_moc_boundsadjusted_occonly_00"

# # view
# feat3d_init = "01_m144x144x144_p128x384_1e-4_O_c1_s1_V_d32_c1_mark_data_train_mark_data_train_ns_enc3d_view_00_fitvox"
# view_init = "01_m144x144x144_p128x384_1e-4_O_c1_s1_V_d32_c1_mark_data_train_mark_data_train_ns_enc3d_view_00_fitvox"
# occ_init = "01_m144x144x144_p128x384_1e-4_O_c1_s1_V_d32_c1_mark_data_train_mark_data_train_ns_enc3d_view_00_fitvox"

# # gqn lesroart
# gqn_init = "32_m128x64x128_1e-4_mark_data_train_mark_data_train_ns_gqn_pool_lescroart04"

det_init = ""

# feat3d_init = "03_m144x144x144_2e-5_O_c1_s.1_ns_carl_rep_lesc02"
# view_init = ""
# occ_init = "03_m144x144x144_2e-5_O_c1_s.1_ns_carl_rep_lesc02"
# total_init = "03_m144x144x144_2e-5_O_c1_s.1_ns_carl_rep_lesc02"