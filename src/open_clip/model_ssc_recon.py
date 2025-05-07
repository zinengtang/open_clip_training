import torch
from typing import Optional
import numpy as np
from torch import nn
from .model import CustomTextCLIP, CLIPVisionCfg, CLIPTextCfg

class SSCReconModel(CustomTextCLIP):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            ssc_cfg: dict = None,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):

        super().__init__(embed_dim, vision_cfg, text_cfg, quick_gelu,
                         init_logit_scale, init_logit_bias, cast_dtype, output_dict)
    
    def forward(
            self,
            data_batches
    ):
        model_out_list = []
        for (modality_0, modality_1) in data_batches:
            if modality_0.dim() > 2 and modality_1.dim() > 2:
                image_0_features = self.encode_image(modality_0, normalize=True, use_ssl_head=True)
                image_1_features = self.encode_image(modality_1, normalize=True, use_ssl_head=True)
                out_dict = {
                    "image_0_features": image_0_features,
                    "text_features": image_1_features,
                    "logit_scale": self.logit_scale_image.exp(),
                }
                if self.logit_bias is not None:
                    out_dict['logit_bias'] = self.logit_bias_image
                model_out_list.append((out_dict, "image"))

            elif modality_0.dim() == 2 and modality_1.dim() == 2:
                image_features = self.encode_text(modality_0, normalize=True, use_ssl_head=True)
                text_features = self.encode_text(modality_1, normalize=True, use_ssl_head=True)
                out_dict = {
                    "image_0_features": image_features,
                    "text_features": text_features,
                    "logit_scale": self.logit_scale_text.exp(),
                }
                if self.logit_bias is not None:
                    out_dict['logit_bias'] = self.logit_bias_text
                model_out_list.append((out_dict, "text"))
        
            else:
                image_features = self.encode_image(modality_0, normalize=True, use_ssl_head=False)
                text_features = self.encode_text(modality_1, normalize=True, use_ssl_head=False)
                out_dict = {
                    "image_0_features": image_features,
                    "text_features": text_features,
                    "logit_scale": self.logit_scale.exp(),
                }
                if self.logit_bias is not None:
                    out_dict['logit_bias'] = self.logit_bias
                model_out_list.append((out_dict, "image-text"))
        
        return model_out_list