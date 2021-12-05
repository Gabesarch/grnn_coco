
import numpy as np
import zarr

output_dir = '/lab_data/tarrlab/gsarch/encoding_model/grnn_feats_nomaxpool'

feats = np.load(output_dir + '/feats.npy')
b,c,h,w,d = feats.shape
feats = np.reshape(feats, (b, -1))

zarr.save(output_dir + '/feats.zarr', feats)