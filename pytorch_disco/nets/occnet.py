import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.pixelshuffle3d
import hyperparams as hyp
import utils.improc
import utils.misc
import utils.basic


import ipdb
st = ipdb.set_trace

class OccNet(nn.Module):
    def __init__(self):
        super(OccNet, self).__init__()

        print('OccNet...')

        # self.conv3d = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=8, kernel_size=1, stride=1, padding=0).cuda()
        # # self.unpack = nn.PixelShuffle(9)
        # self.unpack = archs.pixelshuffle3d.PixelShuffle3d(2)
        
        # self.conv3d = nn.Conv3d(in_channels=int(hyp.feat_dim/2), out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        
        self.conv3d = nn.Conv3d(in_channels=hyp.feat3d_dim, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()

        # self.conv3d = nn.ConvTranspose3d(hyp.feat_dim, 1, kernel_size=4, stride=2, padding=1, bias=False).cuda()
        
    def compute_loss(self, pred, occ, free, valid, summ_writer, set_name=None):
        pos = occ.clone()
        neg = free.clone()

        # occ is B x 1 x Z x Y x X

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        loss_vis = torch.mean(loss*mask_*valid, dim=3)
        # if summ_writer is not None:
        #     front_name = 'occ'
        #     if set_name is not None:
        #         front_name = f'{front_name}_{set_name}'
            # summ_writer.summ_oned(f'{front_name}/prob_loss', loss_vis)
        pos_loss = utils.basic.reduce_masked_mean(loss, pos*valid)
        neg_loss = utils.basic.reduce_masked_mean(loss, neg*valid)

        balanced_loss = pos_loss + neg_loss

        loss_vox = loss*mask_*valid

        return loss_vox, balanced_loss, loss

    def forward(self, feat, occ_g=None, free_g=None, valid=None, summ_writer=None, suffix='', set_name=None):
        total_loss = torch.tensor(0.0).cuda()
        
        occ_e_ = self.conv3d(feat)

        # smooth loss
        if hyp.occ_smooth_coeff>0:
            dz, dy, dx = utils.basic.gradient3d(occ_e_, absolute=True)
            smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
            # if valid is not None:
            #     smooth_loss = utils.basic.reduce_masked_mean(smooth_vox, valid)
            # else:
            smooth_loss = torch.mean(smooth_vox)
            total_loss = utils.misc.add_loss('occ/smooth_loss%s' % suffix, total_loss, smooth_loss, hyp.occ_smooth_coeff, summ_writer)
        
        occ_e_ = occ_e_.squeeze(1)
        occ_e = torch.sigmoid(occ_e_)
        occ_e_binary = torch.round(occ_e)

        if occ_g is not None:
            # assume free_g and valid are also not None
            
            # collect some accuracy stats 
            occ_match = occ_g*torch.eq(occ_e_binary, occ_g).float()
            free_match = free_g*torch.eq(1.0-occ_e_binary, free_g).float()
            either_match = torch.clamp(occ_match+free_match, 0.0, 1.0)
            either_have = torch.clamp(occ_g+free_g, 0.0, 1.0)
            acc_occ = utils.basic.reduce_masked_mean(occ_match, occ_g*valid)
            acc_free = utils.basic.reduce_masked_mean(free_match, free_g*valid)
            acc_total = utils.basic.reduce_masked_mean(either_match, either_have*valid)
            acc_bal = (acc_occ + acc_free)*0.5

            if summ_writer is not None:
                front_name = 'unscaled_occ'
                if set_name is not None:
                    front_name = f'{front_name}_{set_name}'
                summ_writer.summ_scalar(f'{front_name}/acc_occ%s' % suffix, acc_occ.cpu().item())
                summ_writer.summ_scalar(f'{front_name}/acc_free%s' % suffix, acc_free.cpu().item())
                summ_writer.summ_scalar(f'{front_name}/acc_total%s' % suffix, acc_total.cpu().item())
                summ_writer.summ_scalar(f'{front_name}/acc_bal%s' % suffix, acc_bal.cpu().item())

            vox_loss, prob_loss, full_loss = self.compute_loss(occ_e_, occ_g, free_g, valid, summ_writer, set_name)
            total_loss = utils.misc.add_loss('occ/prob_loss%s' % suffix, total_loss, prob_loss, hyp.occ_coeff, summ_writer)
        else:
            full_loss, either_match = None, None

        if summ_writer is not None:
            front_name = 'occ'
            if set_name is not None:
                front_name = f'{front_name}_{set_name}'
            summ_writer.summ_oned(f'{front_name}/smooth_loss%s' % suffix, torch.mean(smooth_vox, dim=3))
            if occ_g is not None:
                summ_writer.summ_occ(f'{front_name}/occ_g%s' % suffix, occ_g)
                summ_writer.summ_oned(f'{front_name}/occ_g_oned%s' % suffix, occ_g, bev=True)
                summ_writer.summ_occ(f'{front_name}/free_g%s' % suffix, free_g)
            summ_writer.summ_occ(f'{front_name}/occ_e%s' % suffix, occ_e.unsqueeze(1))
            summ_writer.summ_occ(f'{front_name}/occ_e_binary%s' % suffix, occ_e_binary.unsqueeze(1))
            summ_writer.summ_oned(f'{front_name}/occ_e_oned%s' % suffix, occ_e.unsqueeze(1), bev=True)
            # summ_writer.summ_occ('occ/valid%s' % suffix, valid)

        return total_loss, occ_e_

