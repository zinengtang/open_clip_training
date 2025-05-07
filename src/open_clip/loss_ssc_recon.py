import torch
import torch.nn.functional as F
from .loss import SigLipLoss, gather_features


class SSCReconLoss(SigLipLoss):
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

        self.siglip_loss_weight = siglip_loss_weight
        self.ssc_loss_weight = ssc_loss_weight
        self.local_loss = local_loss


    def forward(self, image_0_features, image_1_features=None, text_features=None, logit_scale=None, logit_bias=None, logit_scale_ssc=None, output_dict=False):

        siglip_loss, alignment_loss = super().forward(
            image_0_features, text_features, logit_scale, logit_bias)
        return {"ct_loss": siglip_loss, "align_loss": alignment_loss}
