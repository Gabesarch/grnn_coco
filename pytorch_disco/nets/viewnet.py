import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
import ipdb
st = ipdb.set_trace
import archs.encoder3D2D as encoder3D2D
import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_basic
import utils_misc

class ViewNet(nn.Module):
    def __init__(self, feat_dim):
        super(ViewNet, self).__init__()

        print('ViewNet...')

        self.med_dim = 32
        self.net = encoder3D2D.Net3D2D(in_chans=feat_dim, mid_chans=32, out_chans=self.med_dim, depth=hyp.view_depth).cuda()
        self.emb_layer = nn.Conv2d(in_channels=self.med_dim, out_channels=feat_dim, kernel_size=1, stride=1, padding=0).cuda()
        self.rgb_layer = nn.Conv2d(in_channels=self.med_dim, out_channels=3, kernel_size=1, stride=1, padding=0).cuda()

    def forward(self, feat, rgb_g, valid, summ_writer,name,just_return_rgbe=False, set_name=None):
        total_loss = torch.tensor(0.0).cuda()
        front_name = 'view'
        if set_name is not None:
            front_name = f'{front_name}_{set_name}'
        if hyp.dataset_name == "clevr":
            valid = torch.ones_like(valid)
        
        feat = self.net(feat)
        emb_e = self.emb_layer(feat)
        rgb_e = self.rgb_layer(feat)
        # postproc
        emb_e = l2_normalize(emb_e, dim=1)
        rgb_e = torch.tanh(rgb_e)*0.5

        if just_return_rgbe:
            return rgb_e

        loss_im = l1_on_axis(rgb_e-rgb_g, 1, keepdim=True)
        summ_writer.summ_oned(f'{front_name}/rgb_loss', loss_im*valid)
        rgb_loss = utils_basic.reduce_masked_mean(loss_im, valid)

        total_loss = utils_misc.add_loss(f'{front_name}/rgb_l1_loss', total_loss, rgb_loss, hyp.view_l1_coeff, summ_writer)

        # viss
        summ_writer.summ_rgbs(f'{front_name}/{name}', [rgb_e, rgb_g])

        return total_loss, rgb_e, emb_e

