# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.cuda.nvtx as nvtx
import clip
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR

import rvt.utils.peract_utils as peract_utils
import peract_colab.arm.utils as arm_utils
from peract_colab.arm.optim.lamb import Lamb
from yarr.agents.agent import ActResult
from rvt.utils.dataset import _clip_encode_text
from rvt.utils.lr_sched_utils import GradualWarmupScheduler

import rvt.mvt.aug_utils as aug_utils
from rvt.models.eval_utils import *
import rvt.models.pc_utils as pc_utils

class RVTAgent:
    def __init__(
        self,
        network: nn.Module,
        add_lang: bool,
        add_clip: bool=False,
        lr: float = 0.0001,
        lr_cos_dec: bool = False,
        cos_dec_max_step: int = 60000,
        warmup_steps: int = 0,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        place_with_mean: bool = True,
        optimizer_type: str = "lamb",
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = peract_utils.SCENE_BOUNDS,
        log_dir="",
        num_rotation_classes = None,
        transform_augmentation=None,
        transform_augmentation_xyz=None,
        transform_augmentation_rpy=None,
        move_pc_in_bound=None
    ):
        """
        :param gt_hm_sigma: the std of the groundtruth hm, currently for for
            2d, if -1 then only single point is considered
        :type gt_hm_sigma: float
        :param log_dir: a folder location for saving some intermediate data
        """

        self.num_all_rot=216
        self.num_rotation_classes=72
        self._num_rotation_classes=72
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._network = network
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._place_with_mean = place_with_mean
        self._optimizer_type = optimizer_type
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.add_lang = add_lang
        self.log_dir = log_dir
        self.warmup_steps = warmup_steps
        self.lr_cos_dec = lr_cos_dec
        self.cos_dec_max_step = cos_dec_max_step
        self.scene_bounds = scene_bounds
        self.add_clip=add_clip

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        if self._optimizer_type == "lamb":
            # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
            self._optimizer = Lamb(
                self._network.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
                betas=(0.9, 0.999),
                adam=False,
            )
        elif self._optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception("Unknown optimizer")
        
        if self.lr_cos_dec:
            after_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=self.cos_dec_max_step,
                eta_min=self._lr / 1000,  # mininum lr
            )
        else:
            after_scheduler = None
        self._lr_sched = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=after_scheduler,
        )

    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    # copied from per-act and removed the translation part
    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def get_q(self, out, dims, only_pred=False):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        pts = None
        # (bs, h*w, nc)
        # We do not consider wrist image for translation.
        q_trans = out["trans"].view(bs, nc-1, h * w).transpose(1, 2)
        if not only_pred:
            q_trans = q_trans.clone()

        # (bs, 218)
        rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
        grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
        # (bs, 2)
        collision_q = out["feat"].view(bs, -1)[
            :, self.num_all_rot + 2 : self.num_all_rot + 4
        ]
        y_q = None
        return q_trans, rot_q, grip_q, collision_q, y_q, pts
    
    def update(
        self,
        step: int,
        epoch: int,
        replay_sample: dict,
        backprop: bool = True,
        eval_log: bool = False,
        reset_log: bool = False,
    ) -> dict:
        nvtx.range_push(f'step #{step}')
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)
        assert replay_sample["lang_goal_embs"].shape[1:] == (1, 77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (
            1,
            self._net_mod.proprio_dim,
        )
        
        nvtx.range_push(f'preprocess dataset')
        # sample
        action_rot_grip = replay_sample["rot_grip_action_indicies"][
            :, -1
        ].int()  # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][
            :, -1
        ].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        action_grip = action_rot_grip[:, -1]  # (b,)
        lang_goal_embs = replay_sample["lang_goal_embs"][:, -1]
        tasks = replay_sample["tasks"]
        proprio = arm_utils.stack_on_channel(replay_sample["low_dim_state"])  # (b, 4)
        return_out = {}
            
        obs = peract_utils._preprocess_patch_inputs(replay_sample, train=True)
        bs, nc, _, h, w = obs.shape
        if self.add_clip:
            goal_simil = peract_utils._preprocess_goal_simil_inputs(None, None,
                                                                    replay_sample,
                                                                    train=True)
            obs = (obs, goal_simil)
        extrin, intrin = peract_utils._preprocess_input_camerass(replay_sample)
        # print(obs.shape)
        out = self._network(
            img_feat=obs,
            cameras = [extrin, intrin],
            proprio=proprio,
            lang_emb=lang_goal_embs,
            clip_emb=self.add_clip
        )
        nvtx.range_pop()
        nvtx.range_push('forward network')
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )
        nvtx.range_pop()
        nvtx.range_push('get GT')

        nvtx.range_push('get_one_hot_expert_actions')
        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )
        nvtx.range_pop()
        nvtx.range_push('get_action_trans')
        action_trans = pc_utils.get_action_trans(
            action_trans_con, camera = (extrin, intrin), dims=(bs, nc, h, w)
        )

        # if True:
        #     pred, name= pc_utils.get_pred_wpt(action_trans,camera = (extrin, intrin), dims=(bs, nc, h, w))
        #     print("goal is ",action_trans_con)
        #     print("pred is ",pred)
        #     print('-----------')
        #     _ = pc_utils.debug_action_trans(obs, action_trans, action_trans, name,softmax=True)
            # new_image.save(f'/home/guest/Desktop/RVT/tmp/{name}_heatmap.png')
        
        # visualize the translation results
        # if (backprop ==True and step % 20 ==0):
        #     breakpoint()
            
        # visualize the translation results
        if (backprop ==True and step % 1000 ==0) or (backprop==False and step % 100 ==0):
            if self.add_clip:
                new_image = pc_utils.debug_action_trans(obs[0], q_trans, action_trans, patch=True, softmax=True)
            else:
                new_image = pc_utils.debug_action_trans(obs, q_trans, action_trans, patch=True, softmax=True)
        else:
            new_image = None

        nvtx.range_pop()
        nvtx.range_pop()
        nvtx.range_push('Backprop')

        loss_log = {}
        if backprop:
            # cross-entropy loss
            trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            nvtx.range_push('compute loss')
            if self.add_rgc_loss:
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()

                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()
                
                collision_loss = self._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()
            nvtx.range_pop()
                
            total_loss = (
                trans_loss
                + rot_loss_x
                + rot_loss_y
                + rot_loss_z
                + grip_loss
                + collision_loss
            )
            nvtx.range_push('zero grad')
            self._optimizer.zero_grad(set_to_none=True)
            nvtx.range_pop()
            nvtx.range_push('Backward')
            total_loss.backward()
            nvtx.range_pop()
            nvtx.range_push('optimizer step')
            self._optimizer.step()
            nvtx.range_pop()
            nvtx.range_push('schduler step')
            self._lr_sched.step()
            nvtx.range_pop()
            loss_log = {
                "train_total_loss": total_loss.item(),
                "train_trans_loss": trans_loss.item(),
                "train_rot_loss_x": rot_loss_x.item(),
                "train_rot_loss_y": rot_loss_y.item(),
                "train_rot_loss_z": rot_loss_z.item(),
                "train_grip_loss": grip_loss.item(),
                "train_collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "step": step,
            }
            wandb_log = loss_log.copy()
            if new_image is not None:
                images = wandb.Image(new_image)
                wandb_log['train_visualize'] = images
            wandb_log['step'] += wandb_log['epoch'] * 10000
            nvtx.range_push('log')
            wandb.log(wandb_log)
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)
            nvtx.range_pop()

        nvtx.range_pop()
        if eval_log:
            with torch.no_grad():
                wpt = torch.cat([x.unsqueeze(0) for x in action_trans_con])
                pred_wpt, pred_rot_quat, _, _ = self.get_pred(
                        q_trans, rot_q, grip_q, collision_q, \
                        cameras = [extrin, intrin], dims=(bs, nc, h, w), train=False
                )
                return_log = manage_eval_log(
                    self=self,
                    tasks=tasks,
                    trans=wpt,
                    pred_trans=pred_wpt,
                    action_rot=action_rot,
                    pred_rot_quat=pred_rot_quat,
                    action_grip_one_hot=action_grip_one_hot,
                    grip_q=grip_q,
                    action_collision_one_hot=action_collision_one_hot,
                    collision_q=collision_q,
                    reset_log=reset_log,
                )

                if new_image is not None:
                    images = wandb.Image(new_image)
                    return_log['eval_visualize'] = images
                return_out.update(return_log)

        torch.cuda.empty_cache()
        nvtx.range_pop()
        return return_out

    @torch.no_grad()
    def act(
        self, step: int, observation: dict, deterministic=True, pred_distri=False
    ) -> ActResult:
        if self.add_lang:
            lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
            _, lang_goal_embs = _clip_encode_text(self.clip_model, lang_goal_tokens[0])
            lang_goal_embs = lang_goal_embs.float()
        else:
            lang_goal_embs = (
                torch.zeros(observation["lang_goal_embs"].shape)
                .float()
                .to(self._device)
            )
            
        proprio = arm_utils.stack_on_channel(observation["low_dim_state"])
        obs = peract_utils._preprocess_patch_inputs(observation, train=False)
        extrin, intrin = peract_utils._preprocess_input_camerass(observation)
        bs, nc, _, h, w = obs.shape
        out = self._network(
            img_feat=obs,
            cameras = [extrin, intrin],
            proprio=proprio,
            lang_emb=lang_goal_embs,
        )
        
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            q_trans, rot_q, grip_q, collision_q, cameras = [extrin, intrin], dims=(bs, nc, h, w), train=False
        )

        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
            )
        )
        return ActResult(continuous_action)

    def get_pred(
        self,
        q_trans,
        rot_q,
        grip_q,
        collision_q,
        cameras,
        dims,
        train=True,
    ):
        if train==True:
            pred_trans = pc_utils.get_action_trans(q_trans, cameras, dims)
        elif train==False:
            pred_trans = pc_utils.get_pred_wpt(q_trans, cameras, dims)
            
        pred_rot = torch.cat(
            (
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)
        return pred_trans, pred_rot_quat, pred_grip, pred_coll

    def reset(self):
        pass

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()