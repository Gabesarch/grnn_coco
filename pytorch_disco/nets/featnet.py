import torch
import torch.nn as nn

import sys
sys.path.append("..")
import ipdb
st = ipdb.set_trace
import hyperparams as hyp
import archs.encoder3D as encoder3D
from utils.basic import l2_normalize, gradient3d
import utils.misc

class FeatNet(nn.Module):
    def __init__(self, in_dim=4):
        super(FeatNet, self).__init__()
        self.net = encoder3D.Net3D_NOBN(in_channel=in_dim, pred_dim=hyp.feat3d_dim).cuda()

    def forward(self, feat, summ_writer=None, mask=None,prefix=""):
        total_loss = torch.tensor(0.0).cuda()
        B, C, D, H, W = list(feat.shape)
        if summ_writer is not None:
            summ_writer.summ_feat(f'feat3d/{prefix}feat_input', feat, pca=(C>3))
    
        feat = self.net(feat)

        # smooth loss
        dz, dy, dx = gradient3d(feat, absolute=True)
        smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
        smooth_loss = torch.mean(smooth_vox)
        if summ_writer is not None:
            summ_writer.summ_oned(f'feat3d/{prefix}smooth_loss', torch.mean(smooth_vox, dim=3))
        total_loss = utils.misc.add_loss(f'feat3d/{prefix}smooth_loss', total_loss, smooth_loss, hyp.feat3d_smooth_coeff, summ_writer)

        feat = l2_normalize(feat, dim=1)
        if summ_writer is not None:
            summ_writer.summ_feat(f'feat3d/{prefix}feat_output', feat, pca=True)

        return feat, total_loss


