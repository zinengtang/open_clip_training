import torch
from typing import Optional
import numpy as np
from torch import nn
import random
from .model import CustomTextCLIP, CLIPVisionCfg, CLIPTextCfg

from collections import OrderedDict

class SSCModel(CustomTextCLIP):
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
            data_batches,
            sample_index=0,
            get_image_text_reconstruction_loss=False
    ):
        model_out_list = []
        for (modality_0, modality_1) in data_batches:
            if modality_0.dim() > 2 and modality_1.dim() > 2:
                
                if get_image_text_reconstruction_loss:
                    image_0_features, loss_img_image_0 = self.encode_image(image=modality_0, target_image=modality_0, normalize=True, use_ssl_head=True, average_pool=True, sample_index=sample_index, get_image_text_reconstruction_loss=get_image_text_reconstruction_loss)
                    image_1_features = self.encode_image(image=modality_1, normalize=True, use_ssl_head=True, average_pool=True, sample_index=sample_index, get_image_text_reconstruction_loss=False)
                    # print(image_0_features.shape, image_1_features.shape)
                    out_dict = {
                        "image_0_features": image_0_features,
                        "text_features": image_1_features,
                        "logit_scale_image": self.logit_scale_image.exp(),
                        "loss_img_image_0": loss_img_image_0,
                    }
                else:
                    image_0_features = self.encode_image(image=modality_0, normalize=True, average_pool=True, sample_index=sample_index, use_ssl_head=True)
                    image_1_features = self.encode_image(image=modality_1, normalize=True, average_pool=True, sample_index=sample_index, use_ssl_head=True)
                    out_dict = {
                        "image_0_features": image_0_features,
                        "text_features": image_1_features,
                        "logit_scale_image": self.logit_scale_image.exp(),
                    }
                if self.logit_bias_image is not None:
                    out_dict['logit_bias_image'] = self.logit_bias_image
                model_out_list.append((out_dict, "image"))

            elif modality_0.dim() == 2 and modality_1.dim() == 2:
                if get_image_text_reconstruction_loss:
                    image_features, loss_txt_text_0 = self.encode_text(text=modality_0, normalize=True, use_ssl_head=True, get_image_text_reconstruction_loss=True)
                    text_features = self.encode_text(text=modality_1, normalize=True, use_ssl_head=True)
                    out_dict = {
                        "image_0_features": image_features,
                        "text_features": text_features,
                        "logit_scale_text": self.logit_scale_text.exp(),
                        "loss_txt_text_0": loss_txt_text_0,
                    }
                else:
                    image_features = self.encode_text(text=modality_0, normalize=True, use_ssl_head=True)
                    text_features = self.encode_text(text=modality_1, normalize=True, use_ssl_head=True)
                    out_dict = {
                        "image_0_features": image_features,
                        "text_features": text_features,
                        "logit_scale_text": self.logit_scale_text.exp(),
                    }
                if self.logit_bias_text is not None:
                    out_dict['logit_bias_text'] = self.logit_bias_text
                model_out_list.append((out_dict, "text"))
        
            else:
                # if get_image_text_reconstruction_loss:
                #     image_features, loss_img_text = self.encode_image(image=modality_0, text=modality_1, normalize=True, use_ssl_head=False, get_image_text_reconstruction_loss=get_image_text_reconstruction_loss)
                #     text_features = self.encode_text(image=modality_0, text=modality_1, normalize=True, use_ssl_head=False, get_image_text_reconstruction_loss=get_image_text_reconstruction_loss)
                #     # print(image_features.shape, text_features.shape)
                #     out_dict = {
                #         "image_0_features": image_features,
                #         "text_features": text_features,
                #         "logit_scale": self.logit_scale.exp(),
                #         "loss_img_text": loss_img_text,
                #     }
                # else:
                image_features = self.encode_image(image=modality_0, normalize=True, use_ssl_head=False)
                text_features = self.encode_text(text=modality_1, normalize=True, use_ssl_head=False)
                out_dict = {
                    "image_0_features": image_features,
                    "text_features": text_features,
                    "logit_scale": self.logit_scale.exp(),
                }
                if self.logit_bias is not None:
                    out_dict['logit_bias'] = self.logit_bias
                model_out_list.append((out_dict, "image-text"))
        
        return model_out_list