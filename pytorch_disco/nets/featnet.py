import torch
import torch.nn as nn

import sys
sys.path.append("..")
import ipdb
st = ipdb.set_trace
import hyperparams as hyp
import archs.encoder3D as encoder3D
from utils_basic import l2_normalize

class FeatNet(nn.Module):
    def __init__(self, in_dim=4):
        super(FeatNet, self).__init__()
        self.net = encoder3D.Net3D_NOBN(in_channel=in_dim, pred_dim=hyp.feat_dim).cuda()

    def forward(self, feat, summ_writer=None, mask=None,prefix=""):
        total_loss = torch.tensor(0.0).cuda()
        B, C, D, H, W = list(feat.shape)
        if summ_writer is not None:
            summ_writer.summ_feat(f'feat/{prefix}feat_input', feat)
    
        feat = self.net(feat)
        feat = l2_normalize(feat, dim=1)
        if summ_writer is not None:
            summ_writer.summ_feat(f'feat/feat_output', feat)

        return feat,  total_loss


