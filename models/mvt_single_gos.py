# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from math import ceil

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch import nn
from einops import rearrange, repeat
from rvt.mvt.modules_gos import *
from rvt.mvt.attn import (
    Conv2DBlock,Conv2DUpsampleBlock_org,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    cache_fn,
    DenseBlock,
    FeedForward,
)

MAX_TOKEN=20

class MVT(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        add_proprio,
        proprio_dim,
        add_lang,
        lang_dim,
        lang_len,
        img_feat_dim,
        feat_dim,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        add_pixel_loc,
        add_depth,
        pe_fix,
        renderer_device="cuda:0",
        renderer=None,
    ):
        """MultiView Transfomer

        :param depth: depth of the attention network
        :param img_size: number of pixels per side for rendering
        :param renderer_device: device for placing the renderer
        :param add_proprio:
        :param proprio_dim:
        :param add_lang:
        :param lang_dim:
        :param lang_len:
        :param img_feat_dim:
        :param feat_dim:
        :param im_channels: intermediate channel size
        :param attn_dim:
        :param attn_heads:
        :param attn_dim_head:
        :param activation:
        :param weight_tie_layers:
        :param attn_dropout:
        :param decoder_dropout:
        :param img_patch_size: intial patch size
        :param final_dim: final dimensions of features
        :param self_cross_ver:
        :param add_corr:
        :param add_pixel_loc:
        :param add_depth:
        :param pe_fix: matter only when add_lang is True
            Either:
                True: use position embedding only for image tokens
                False: use position embedding for lang and image token
        """

        super().__init__()
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        self.lang_dim = lang_dim
        self.lang_len = lang_len
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.pe_fix = pe_fix

        print(f"MVT Vars: {vars(self)}")

        self.renderer = renderer
        self.num_img = 6

        # patchified input dimensions
        spatial_size = img_size // self.img_patch_size  # 128 / 8 = 16

        if self.add_proprio:
            # 64 img features + 64 proprio features
            self.input_dim_before_seq = int(self.im_channels * 5/2)
        else:
            self.input_dim_before_seq = self.im_channels

        # learnable positional encoding
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        if self.pe_fix:
            num_pe_token = spatial_size**2 * (self.num_img * MAX_TOKEN)
        else:
            num_pe_token = lang_max_seq_len + (spatial_size**2 * self.num_img)
        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.input_dim_before_seq,
            )
        )
        self.final_dim = self.final_dim // 2
        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )

        # img input preprocessing encoder
        self.input_preprocess = Conv2DBlock(
            inp_img_feat_dim + 2,
            self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=activation,
        )
        inp_pre_out_dim = self.im_channels
        self.extrinsic_preprocess = DenseBlock(
            16,
            self.im_channels // 2,
            norm="group",
            activation=activation,
        )

        self.intrinsic_preprocess = DenseBlock(
            9,
            self.im_channels // 2,
            norm="group",
            activation=activation,
        )
        self.proprio_preprocess = DenseBlock(
            self.proprio_dim,
            self.im_channels // 2,
            norm="group",
            activation=activation,
        )

        self.patchify = Conv2DBlock(
            inp_pre_out_dim,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )

        # lang preprocess
        self.lang_preprocess = DenseBlock(
            lang_emb_dim * lang_max_seq_len,
            attn_dim,
            norm="group",
            activation=activation,
        )

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            attn_dim,
            norm=None,
            activation=None,
        )

        self.fc_during_attn = DenseBlock(
            attn_dim * spatial_size * spatial_size,
            attn_dim,
            norm=None,
            activation=None,
        )

        self.fc_aft_attn = DenseBlock(
            attn_dim*4,
            self.input_dim_before_seq * spatial_size * spatial_size,
            norm="group",
            activation=activation,
        )

        get_attn_attn = lambda: PreNorm(
            attn_dim,
            Attention(
                attn_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
            ),
        )
        
        get_attn_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))
        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        attn_depth = depth[-2]

        for _ in range(attn_depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)])
            )

        self.obj_encoder = TransformerEncoder_gos_and_masking(
            input_dim=attn_dim,
            num_layers=int(depth[-1]-depth[-2]),
            num_heads=attn_heads,
            dim_head_output=attn_dim_head,
            mlp_dim=attn_dim,
            dropout=attn_dropout,
            prune_num ={'0':0, '1':2, '2':0, '3':1, '4':0}, # ours
            keep_num= {'0':10, '1':8, '2':6, '3':5, '4':4},
            # prune_num ={'0':0, '1':2, '2':0, '3':2, '4':0}, # abla3 --- not doing this (no time).
            # keep_num= {'0':12, '1':10, '2':8, '3':6, '4':4},
            # prune_num ={'0':8, '1':2, '2':2, '3':1, '4':1}, # abla2: only pruning
            # keep_num= {'0':12, '1':10, '2':8, '3':7, '4':6},
            # prune_num ={'0':0, '1':0, '2':0, '3':0, '4':0}, # abla1: only merging
            # keep_num= {'0':10, '1':8, '2':6, '3':5, '4':4},
        )

        UPSAMPLE_SPLIT = 4
        self.up0 = Conv2DUpsampleBlock(
            self.input_dim_before_seq,
            self.im_channels,
            kernel_sizes=int(self.img_patch_size/UPSAMPLE_SPLIT),
            strides=int(self.img_patch_size/UPSAMPLE_SPLIT),
            norm="group",
            activation=activation,
        )
        self.up1 = Conv2DUpsampleBlock(
            self.im_channels,
            self.im_channels,
            kernel_sizes=UPSAMPLE_SPLIT,
            strides=UPSAMPLE_SPLIT,
            norm="group",
            activation=activation,
        )

        final_inp_dim = self.im_channels  * 2
        # final_inp_dim = self.im_channels  * (1+MAX_TOKEN)

        # final layers
        self.final = Conv2DBlock(
            final_inp_dim,
            self.final_dim,
            kernel_sizes=3,
            strides=1,
            norm="group",
            activation=activation,
        )
        self.trans_decoder = Conv2DBlock(
            self.final_dim,
            1,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=None,
        )
        feat_out_size = feat_dim
        feat_fc_dim = 0
        feat_fc_dim += self.input_dim_before_seq
        feat_fc_dim += self.final_dim

        self.feat_fc = nn.Sequential(
            nn.Linear(self.num_img * feat_fc_dim, feat_fc_dim),
            nn.ReLU(),
            nn.Linear(feat_fc_dim, feat_fc_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_fc_dim // 2, feat_out_size),
        )



    def forward(
        self,
        img,
        bbox = None,
        cameras = None,
        proprio=None,
        lang_emb=None,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        """
        rgb, mask = img
        bs, num_img, num_mask, img_feat_dim, h, w = mask.shape
        num_pat_img = h // self.img_patch_size
        assert num_img == self.num_img
        rgb = rgb.unsqueeze(2).repeat(1,1,num_mask,1,1,1)
        img = torch.cat((rgb, mask), dim=3)
        # img_feat_dim = 3 (img channel) + 1 (depth channel) + 1 (mask channel)
        img = img.view(bs * num_img * num_mask, -1, h, w)
        # preprocess
        # img: (bs * num_img * num_mask , img_feat_dim, h, w) 
        #    --> d0: (bs * num_img * num_mask , im_channels, h, w)
        d0 = self.input_preprocess(img)
        imgx = self.patchify(d0)
        imgx = imgx.reshape(bs, num_img, num_mask, imgx.shape[1], num_pat_img, num_pat_img)
        imgx = rearrange(imgx, "b i m c h w -> b i m h w c")
        
        # extrinsic camera info - different by each img
        # cameras: (bs, num_img, 4,4) --> (bs, num_img, num_mask, h, w,img_feat_dim)
        extrin, intrin = cameras                                              
        extrinsic = extrin.flatten(-2,-1).flatten(0,1)
        extrinsic = self.extrinsic_preprocess(extrinsic)              
        extrinsic = extrinsic.view(bs, num_img, -1)
        extrinsic = extrinsic[:,:,None, None, None, :].repeat(1, 1, num_mask, num_pat_img, num_pat_img, 1)

        # intrinsic camera info - different by each img
        # cameras: (bs, num_img, 3,3) --> (bs, num_img, num_mask, h, w,img_feat_dim)
        intrinsic = intrin.flatten(-2,-1).flatten(0,1)
        intrinsic = self.intrinsic_preprocess(intrinsic)          
        intrinsic = intrinsic.view(bs, num_img, -1)
        intrinsic = intrinsic[:,:,None, None, None, :].repeat(1, 1, num_mask, num_pat_img, num_pat_img, 1)
        
        # proprio info           - different by each batch
        # cameras: (bs,4) --> (bs, num_img, num_mask, h, w,img_feat_dim)
        p = self.proprio_preprocess(proprio)
        p = p.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        p = p.repeat(1, num_img, num_mask, num_pat_img,num_pat_img,1)
        imgx = torch.cat([imgx, extrinsic, intrinsic, p], dim=-1)
        
        # add positional encoding
        imgx = rearrange(imgx, "b i m h w c -> b (i m h w) c")
        imgx += self.pos_encoding

        # Across mask patches
        # Work in (bs * num_img * num_mask, h*w, img_feat_dim)
        imgx = self.fc_bef_attn(imgx)
        imgx = imgx.reshape(bs, num_img, num_mask, num_pat_img, num_pat_img, -1)
        imgx = rearrange(imgx,"b i m h w c -> (b i m) (h w) c")
        for self_attn, self_ff in self.layers[:self.depth[0]]:
            imgx = self_attn(imgx) + imgx
            imgx = self_ff(imgx) + imgx

        # Across masks
        # Work in (bs * num_img * num_mask, h*w*img_feat_dim)
        imgx = imgx.reshape(bs, num_img, num_mask, num_pat_img, num_pat_img, -1)
        imgx = rearrange(imgx, "b i m h w c -> (b i) m (h w c)")
        imgx = self.fc_during_attn(imgx)
        for self_attn, self_ff in self.layers[self.depth[0]:self.depth[1]]:
            imgx = self_attn(imgx) + imgx
            imgx = self_ff(imgx) + imgx
            
        # Across all image masks
        # Work in (bs , num_img * num_mask, h*w*img_feat_dim)
        imgx = imgx.reshape(bs, num_img, num_mask, -1)
        imgx = rearrange(imgx, "b i m hwc -> b (i m) hwc")
        for self_attn, self_ff in self.layers[self.depth[1]:self.depth[2]]:
            imgx = self_attn(imgx) + imgx
            imgx = self_ff(imgx) + imgx
            
        # prepare language info
        lx = self.lang_preprocess(lang_emb.flatten(1,-1)).unsqueeze(1)
        lx = lx.repeat(num_img, 1, 1)

        # Across masks
        imgx = imgx.reshape(bs, num_img, num_mask, -1)
        imgx = rearrange(imgx, "b i m hwc -> (b i) m hwc")
        x = torch.cat((lx, imgx), dim=1)
        x = self.obj_encoder(x)
        
        imgx = x[:, 1:,:]
        imgx = imgx.reshape(bs * num_img, -1)
        imgx = self.fc_aft_attn(imgx)
        imgx = imgx.reshape(bs, num_img, num_pat_img, num_pat_img, self.input_dim_before_seq) 
        imgx = rearrange(imgx, "b i p1 p2 d -> b i d p1 p2")  # [B, num_img, 320, num_mask, np, np]
        
        feat = []
        _feat = torch.max(torch.max(imgx, dim=-1)[0], dim=-1)[0]
        _feat = _feat.reshape(bs, -1)
        feat.append(_feat)
        imgx= rearrange(imgx, "b i d p1 p2 -> (b i) d p1 p2")

        u0 = self.up0(imgx)
        u0 = self.up1(u0)

        d0 = d0.reshape(bs, self.num_img, num_mask, -1, h, w)
        # d0_filtered=[]
        # for batch_i in range(bs):
        #     for img_i in range(self.num_img):
        #         idxx = batch_i *bs+img_i
        #         d0_filtered.append(d0[batch_i][img_i][list(survived_idx[idxx])].mean(dim=0))
        # d0_filtered = torch.stack(d0_filtered)
        # u0 = torch.cat([u0, d0_filtered], dim=1)
        d0 = d0.reshape(bs* self.num_img, num_mask, -1, h, w)
        d0 = d0.mean(dim=1)
        u0 = torch.cat([u0, d0], dim=1)
        u = self.final(u0)

        # translation decoder
        trans = self.trans_decoder(u).reshape(bs, self.num_img, h, w) 
        hm = F.softmax(trans.detach().reshape(bs, self.num_img, h * w), 2).reshape(
            bs * self.num_img, 1, h, w
        )

        _feat = torch.sum(hm * u, dim=[2, 3])
        _feat = _feat.reshape(bs, -1)
        
        feat.append(_feat)
        feat = torch.cat(feat, dim=-1)
        feat = self.feat_fc(feat)
        
        # We do not consider wrist image for translation.
        trans = trans[:,:-1,:,:]
        out = {"trans": trans, "feat": feat}

        return out

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        torch.cuda.empty_cache()