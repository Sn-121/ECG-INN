import logging
from collections import OrderedDict
import torchvision.transforms as transformer
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import os
from torch.nn.parallel import DataParallel, DistributedDataParallel
from MainNet.models.archs.discrimantor import NLayerDiscriminator

import MainNet.models.networks as networks
import MainNet.models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from MainNet.models.loss import CharbonnierLoss,cross_entropy_loss_RCF
from MainNet.models.loss_new import SSIMLoss,VGGLoss,GradientLoss,LabLoss
import torch.nn.functional as F
import random
from ..metrics.calculate_PSNR_SSIM import psnr_np
logger = logging.getLogger('base')


class SIEN_Model(BaseModel):
    def __init__(self, opt):
        super(SIEN_Model, self).__init__(opt)

        self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # Initialize discriminator
        # if self.opt['train']['use_discriminator'] :
        #     self.discriminator_style = NLayerDiscriminator(input_nc=3, ndf=64,n_layers=3,use_sigmoid='store_true',gpu_ids=['0'])
        #     self.discriminator_style_optimizer = torch.optim.Adam(list(self.discriminator_style.parameters()),
        #                                                           lr=0.0001,betas=(0.5, 0.999))

        self.load()
####################################################################
        if self.is_train:
            self.netG.train()
            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                self.mse = nn.MSELoss().to(self.device)
                self.cri_grad = GradientLoss().to(self.device)
                self.cri_lab = LabLoss().to(self.device)
                self.cri_vgg = VGGLoss(id=4).to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                # self.cri_vgg = VGGLoss(id=4).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))


            self.l_pix_w = train_opt['pixel_weight']
            self.l_ssim_w = train_opt['ssim_weight']
            self.l_vgg_w = train_opt['vgg_weight']

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['fix_some_part']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        LQ_IMG = data['LQ']
        GT_IMG = data['GT']
        # edge_img = data['EDGE']

        nocropin=data['nocropin']
        nocropgt = data['nocropgt']

        # LQright_IMG = data['LQright']
        # transform = transformer.Resize((256,256))
        # LQ_IMG=transform(LQ_IMG)
        # GT_IMG = transform(GT_IMG)

        self.nocropin = nocropin.to(self.device)
        self.nocropgt = nocropgt.to(self.device)

        self.var_L = LQ_IMG.to(self.device)
        # self.edge = edge_img.to(self.device)
        # self.varright_L = LQright_IMG.to(self.device)
        if need_GT:
            self.real_H = GT_IMG.to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0


    def optimize_parameters(self, step):
        if self.opt['train']['fix_some_part'] and step < self.opt['train']['fix_some_part']:
            self.set_params_lr_zero()

        self.netG.zero_grad() ################################################# new add
        self.optimizer_G.zero_grad()

        RL, LL, edge = self.netG(self.var_L.detach(),self.real_H.detach(), rev=False)
        if self.opt['train']['dual']:
            RR_rev, LL_rev= self.netG(self.var_L.detach(),self.real_H.detach(), rev=True)

        # if self.opt['train']['use_discriminator']:
        #     self.train_discriminator_img(self.real_H.detach(),step)


        l_total = self.cri_pix(RL*LL, self.real_H)+(1-self.cri_ssim(RL*LL, self.real_H))\
                  +self.cri_vgg(RL*LL, self.real_H)
                  # +self.cri_pix(LL, LL_rev)\+self.cri_pix(RL ,RR_rev)+self.cri_lab(RL*LL,self.real_H)


        # loss_color = cri_pix(color_out,self.real_H/(torch.mean(self.real_H,1).unsqueeze(1)+1e-8))
        # loss_edge = cross_entropy_loss_RCF(edge_out, canny_out, 1.1) * 5.0

        l_a = l_total

        if self.opt['train']['dual']:
            l_rev = self.cri_pix(RR_rev*LL_rev, self.var_L.detach())
            l_total += 0.1*l_rev

        l_total.backward()
        self.optimizer_G.step()
        self.fake_H = RL*LL
        psnr = psnr_np(self.fake_H.detach(), self.real_H.detach())
        if self.opt['train']['dual']:
            psnr_rev = psnr_np(RR_rev*LL_rev, self.var_L.detach())
        else:
            psnr_rev = psnr
        # set log
        self.log_dict['psnr'] = psnr.item()
        self.log_dict['psnr_rev'] = psnr_rev.item() ### psnr_rev.item()
        self.log_dict['l_total'] = l_total.item()
        self.log_dict['l_a'] = l_a.item()
        self.log_dict['l_rev'] = l_rev.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():

            R,L,_= self.netG(self.nocropin,self.nocropgt,rev = False)

            self.fake_H = R*L
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.nocropin.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.nocropgt.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])


    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def save_best_psnr(self,name):
        self.save_network(self.netG, 'bestpsnr'+name, 0)

    def save_best_ssim(self, name):
        self.save_network(self.netG, 'bestssim' + name, 0)

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
    @staticmethod
    def discriminator_img_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()
        return real_loss + fake_loss
    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty
    def train_discriminator_img(self, real,step):
        loss_dict = {}
        self.requires_grad(self.discriminator_style, True)

        with torch.no_grad():
            R,L= self.netG(self.var_L,self.edge, self.real_H.detach(),rev = False)
            fake = R*L

        real_pred = self.discriminator_style(real)
        fake_pred = self.discriminator_style(fake)
        loss = self.discriminator_img_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_img_loss'] = float(loss)

        self.discriminator_style_optimizer.zero_grad()
        loss.backward()
        self.discriminator_style_optimizer.step()

        # r1 regularization
        d_regularize = step % 16 == 0
        if d_regularize:
            real_w = real.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator_style(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            self.discriminator_style.zero_grad()
            r1_final_loss = 5 * r1_loss * 16 + 0 * real_pred[0].mean()
            r1_final_loss.backward()
            self.discriminator_style_optimizer.step()
            loss_dict['discriminator_img_r1_loss'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator_style, False)

        return loss_dict
    # color_gt = F.avg_pool2d(self.real_H, 3, 1, 1)
    # color_gt = color_gt / torch.sum(color_gt, 1, keepdim=True)
    # # torchvision.utils.save_image(enhanced_img, save_img_path)
    # color_loss = (color_gt-out).abs().mean()
    #
    # l_total+=color_loss