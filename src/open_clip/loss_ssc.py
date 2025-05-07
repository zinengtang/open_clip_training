import torch
import torch.nn.functional as F
from .loss import SigLipLoss, gather_features


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center.to(teacher_output)) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)
            _t = _t.to(self.center)
            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True

class SSCLoss(SigLipLoss):
    def __init__(
            self,
            siglip_loss_weight,
            ssc_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            rank=0,
            world_size=1,
    ):
        super().__init__(
            rank=rank,
            world_size=world_size,
        )

        dino_dim = 256
        self.dino_loss = DINOLoss(dino_dim)

        self.siglip_loss_weight = siglip_loss_weight
        self.ssc_loss_weight = ssc_loss_weight
        self.local_loss = local_loss

        self.teacher_temp = 0.07
        self.n_global_crops_teacher = 1
        self.n_local_crops = 1
        self.n_local_crops_loss_terms = 1

        embed_dim = 1152

    def forward(self, image_0_features, image_1_features=None, text_features=None, logit_scale=None, logit_bias=None, logit_scale_ssc=None, exchange=False, use_hn=False, output_dict=False):

        # if image_0_features.dim() > 2 and text_features.dim() > 2:

        #     teacher_features = image_0_features
        #     student_features = text_features
        #     print(teacher_features.shape)

        #     teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
        #         teacher_features, teacher_temp=self.teacher_temp
        #     ).view(self.n_global_crops_teacher, -1, *image_0_features.shape[1:])
        #     self.dino_loss.update_center(student_features)

        #     loss = self.dino_loss(
        #         student_output_list=student_features.chunk(self.n_local_crops),
        #         teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
        #     ) / self.n_local_crops_loss_terms

        # else:
        loss, _ = super().forward(
            image_0_features, text_features, logit_scale, logit_bias, use_hn=use_hn, exchange=exchange)
        return loss
