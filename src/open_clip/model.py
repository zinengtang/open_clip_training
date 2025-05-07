""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from transformers.modeling_outputs import BaseModelOutputWithPooling
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer, \
    text_global_pool
# from timm.models.vision_transformer import global_pool_nlc
from transformers import AutoModel, AutoModelForCausalLM
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    visual_feature_dim: int = 1024

    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    patch_dropout: float = 0.
    # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attentional_pool: bool = False
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    # a valid model name overrides layers, width, patch_size
    timm_model_name: Optional[str] = None
    # use (imagenet) pretrained weights for named model
    timm_model_pretrained: bool = False
    # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_pool: str = 'avg'
    # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj: str = 'linear'
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth
    dino_model_name: Optional[str] = None
    hf_model_name: Optional[str] = None


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.hf_model_name:
        visual = AutoModel.from_pretrained(
            vision_cfg.hf_model_name,
            trust_remote_code=True,
        )
        embed_dim = visual.config.hidden_size
    elif vision_cfg.dino_model_name:
        visual = AutoModel.from_pretrained(vision_cfg.dino_model_name)
        embed_dim = vision_cfg.visual_feature_dim
    elif vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (
            torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual, embed_dim


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = AutoModelForCausalLM.from_pretrained(
            text_cfg.hf_model_name,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            token="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",
        )
        embed_dim = text.config.hidden_size
        # text = HFTextEncoder(
        #     text_cfg.hf_model_name,
        #     output_dim=embed_dim,
        #     proj_type=text_cfg.hf_proj_type,
        #     pooler_type=text_cfg.hf_pooler_type,
        #     pretrained=text_cfg.hf_model_pretrained,
        #     output_tokens=text_cfg.output_tokens,
        # )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (
            torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer( 
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        embed_dim = text_cfg.width
    return text, embed_dim


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual, embed_dim = _build_vision_tower(
            embed_dim, vision_cfg, quick_gelu, cast_dtype)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = 96
        # text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups,
                         freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(
            cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(
            image, normalize=True) if image is not None else None
        text_features = self.encode_text(
            text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()

from transformers.activations import ACT2FN

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act='silu'):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MultiheadAttentionBlock(nn.Module):
    """Single block: (Probe -> MultiheadAttention -> LayerNorm -> MLP -> Residual)."""
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, layer_norm_eps=1e-6):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            batch_first=True
        )
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

    def forward(self, probe, hidden_state, key_padding_mask=None):
        # Multi-head attention, using probe as query
        hidden_state = self.input_layernorm(hidden_state)
        attn_out, _ = self.attention(
            query=probe,
            key=hidden_state,
            value=hidden_state,
            key_padding_mask=key_padding_mask
        )
        
        # First residual connection
        residual = attn_out
        # LayerNorm + MLP
        attn_out = self.layernorm(attn_out)
        attn_out = residual + self.mlp(attn_out)
        
        return attn_out

class MultiheadAttentionPoolingHead(nn.Module):
    """Multi-layer Multihead Attention Pooling with attention mask support."""
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        output_dim,
        probe_size=1,
        out_probe_size=1,
        layer_norm_eps=1e-5,
        num_layers=2,  # number of attention–MLP blocks
    ):
        super().__init__()
        # Probe for the query. Shape: (1, probe_size, hidden_size)
        self.probe = nn.Parameter(torch.zeros(1, probe_size, hidden_size))
        self.out_probe = nn.Parameter(torch.zeros(1, out_probe_size, hidden_size))
        self.num_layers = num_layers
        
        # Create multiple layers (blocks)
        self.layers = nn.ModuleList([
            MultiheadAttentionBlock(
                hidden_size, 
                intermediate_size, 
                num_attention_heads, 
                layer_norm_eps
            ) 
            for _ in range(num_layers)
        ])
        
        # Final projection
        self.projection = nn.Linear(hidden_size, output_dim, bias=True)

    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.shape[0]
        
        # Prepare the probe tensor for the entire batch
        if self.probe.shape[1] != hidden_state.shape[1]:
            probe = self.probe.repeat(batch_size, 1, 1)
        else:
            probe = hidden_state
        if self.out_probe.shape[1] != hidden_state.shape[1]:
            out_probe = self.out_probe.repeat(batch_size, 1, 1)
        else:
            out_probe = hidden_state
        
        # Prepare key_padding_mask if attention_mask is given
        # MultiheadAttention expects True at positions to be masked (i.e. "key_padding_mask")
        key_padding_mask = (~attention_mask.bool()) if (attention_mask is not None) else None
        
        # Pass the probe & hidden_state through multiple layers
        for i, block in enumerate(self.layers):
            if i == 0:
                probe = block(probe, hidden_state, key_padding_mask=key_padding_mask)
            elif i == self.num_layers - 1:
                probe = block(out_probe, probe, key_padding_mask=key_padding_mask)
            else:
                probe = block(probe, probe)
        
        # Finally, project the probe output
        return self.projection(probe)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class PatchEmbed(nn.Module):
    """Image to Patch Embedding similar to ViT."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.embed_dim = embed_dim

        # Convolutional layer to create patch embeddings
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        x = self.proj(x)  # [batch_size, embed_dim, grid_size, grid_size]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x

class ImageTextDecoderModel(nn.Module):
    """A model that integrates features, image embeddings, and text embeddings into a decoder."""
    def __init__(
        self,
        features_dim=768,
    ):
        super().__init__()

        # from peft import LoraConfig, get_peft_model
        # # lora_config = LoraConfig(
        # #     r=8,  # Rank of the LoRA matrices
        # #     lora_alpha=32,  # Scaling factor
        # #     target_modules=["q_proj", "k_proj", "v_proj"],  # Modules to apply LoRA
        # #     lora_dropout=0.1,
        # #     bias="none",
        # # )
        # # Apply LoRA to the model
        # # Load the decoder model
        # self.decoder_model = AutoModelForCausalLM.from_pretrained(
        #     decoder_model_name,
        #     torch_dtype="auto",
        #     trust_remote_code=True,
        # )
        # # self.decoder_model = get_peft_model(self.decoder_model, lora_config)
        # self.decoder_config = self.decoder_model.config
        # self.decoder_hidden_size = self.decoder_config.hidden_size
        # # Projection layer for features
        # # self.features_proj_text = nn.Linear(features_dim, self.decoder_hidden_size)
        # # Tokenizer for the decoder model
        # self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name, token="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", trust_remote_code=True)
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
        # from diffusers import StableDiffusion3Pipeline
        # import torch

        # model_id = "stabilityai/stable-diffusion-3.5-medium"
        # self.transformer = SD3Transformer2DModel.from_pretrained(
        #     model_id,
        #     subfolder="transformer",
        #     torch_dtype=torch.bfloat16,
        #     init_lora_weights="gaussian",
        #     token='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',
        # )
        # transformer_lora_config = LoraConfig(
        #     r=8,  # Rank of the LoRA matrices
        #     lora_alpha=32,  # Scaling factor
        #     target_modules=["to_q", "to_k", "to_v"],  # Modules to apply LoRA
        #     lora_dropout=0.1,
        #     bias="none",
        # )
        # self.transformer.add_adapter(transformer_lora_config)
        # for name, param in self.transformer.named_parameters():
        #     if "lora" in name:  # or some other sub-module
        #         param.requires_grad = True
        # self.transformer = get_peft_model(self.transformer, transformer_lora_config)
        # for p in self.transformer.parameters(): p.requires_grad_(True)
        # from diffusers import AutoencoderKL
        # self.vae = AutoencoderKL.from_pretrained(
        #         model_id,
        #         subfolder="vae",
        #         token='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',
        #     )
        # from diffusers import FlowMatchEulerDiscreteScheduler
        # import copy
        # self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        #         model_id, 
        #         subfolder="scheduler",
        #         token='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',
        #     )
        # self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        # self.features_proj_vision = nn.Linear(4096, 4096)
        

        # self.negative_embed = nn.Parameter(torch.randn([1, probe_size, 4096]))
        # self.negative_embed_pooled = nn.Parameter(torch.randn([1, 2048]))
        # import torch
        # from models import build_vae_var
        # MODEL_DEPTH = 16
        # assert MODEL_DEPTH in {16, 20, 24, 30}
        # # download checkpoint
        # vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
        # patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # if 'vae' not in globals() or 'var' not in globals():
        #     self.vae, self.var = build_vae_var(
        #         V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        #         device=device, patch_nums=patch_nums,
        #         num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        #     )
        # # load checkpoints
        # self.vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
        # self.var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
        # # self.vae.eval(), self.var.eval()
        # # self.vae, self.var = self.vae.to(torch.bfloat16), self.var.to(torch.bfloat16)
        # for p in self.vae.parameters(): p.requires_grad_(False)
        # for p in self.var.parameters(): p.requires_grad_(True)
        # lora_config = LoraConfig(
        #     r=16,  # Rank of the LoRA matrices
        #     lora_alpha=64,  # Scaling factor
        #     target_modules=["mat_qkv"],  # Modules to apply LoRA
        #     lora_dropout=0.1,
        #     bias="none",
        # )
        # self.var = self.var.float()
        # self.var.is_float32 = False
        # # self.var = get_peft_model(self.var, lora_config)
        # self.features_proj_vision = nn.Linear(features_dim, 1024)

        # Initialize weights
        # self._init_weights()

        from .decoders.mae import ViTMAEForPreTraining, ViTMAEConfig
        config = ViTMAEConfig.from_pretrained('facebook/vit-mae-base')
        config.decoder_hidden_size = features_dim
        config.hidden_size = config.decoder_hidden_size
        config.decoder_intermediate_size = config.decoder_hidden_size * 4
        config.decoder_num_attention_heads = config.num_attention_heads = 8 if features_dim == 512 else 12
        config.decoder_num_hidden_layers = 12
        self.model = ViTMAEForPreTraining(config)
        del self.model.vit
        self.image_decoder = self.model.decoder

        # pip install accelerate
        from transformers import T5Tokenizer
        from .decoders.t5 import T5ForConditionalGeneration, T5Config
        config = T5Config.from_pretrained("google/t5-efficient-small")
        config.d_model = features_dim
        config.d_ff = config.d_model * 3
        config.num_heads = 8 if features_dim == 512 else 12
        config.num_layers = 12
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-efficient-small")
        self.text_decoder = T5ForConditionalGeneration(config)
        del self.text_decoder.encoder

    def _init_weights(self):
        nn.init.xavier_uniform_(self.features_proj_text.weight)
        if self.features_proj_text.bias is not None:
            nn.init.zeros_(self.features_proj_text.bias)

        nn.init.xavier_uniform_(self.features_proj_vision.weight)
        if self.features_proj_vision.bias is not None:
            nn.init.zeros_(self.features_proj_vision.bias)

    def compute_loss(self, features_vision, features_text, image, raw_text):
        
        if features_vision is not None:
            # features_vision = self.features_proj_vision(image_features)
            decoder_outputs = self.image_decoder(features_vision)
            logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
            pixel_values = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            pixel_values = pixel_values * 0.5 + 0.5
            # logging.info('pixel values', pixel_values.max(), pixel_values.min())
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pixel_values)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pixel_values)
            pixel_values = (pixel_values - mean) / std
            recon_loss = self.model.forward_loss(pixel_values, logits).mean() * 0.1

        elif features_text is not None:
            # features_text = self.features_proj_text(image_features)
            # Project features to decoder's hidden size
            text_input_ids = raw_text
            raw_text = ['<extra_id_0>'+item.replace('</s>', '').replace('<s>', '').strip() for item in raw_text]
            text_input_ids = self.tokenizer.batch_encode_plus(
                raw_text, 
                return_tensors='pt',
                max_length=64,
                padding='max_length',
                padding_side='right',
                truncation=True,
            ).input_ids.to(features_text.device)
            decoder_attention_mask = text_input_ids!=self.tokenizer.pad_token_id
            outputs = self.text_decoder(
                encoder_outputs=[features_text], 
                decoder_input_ids=text_input_ids, 
                decoder_attention_mask=decoder_attention_mask
            )
            text_logits = outputs.logits
            # Prepare labels for loss calculation (teacher forcing)
            shifted_labels = text_input_ids[:, 1:].contiguous()  # [batch_size, seq_len_text - 1]
            shifted_logits = text_logits[:, :-1, :].contiguous()  # [batch_size, seq_len_text - 1, vocab_size]
            # Compute text prediction loss (CrossEntropyLoss)
            text_loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            recon_loss = text_loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)) * 0.1
        # features_vision = self.features_proj_vision(features)
        # import torch.nn as nn
        # train_loss = nn.CrossEntropyLoss(reduction='none')

        # mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(image)
        # std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(image)
        # image_tensor_denorm = image * std + mean
        # image_tensor_resized = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
        # pixel_values = image_tensor_resized * 2 - 1
        # pixel_values = image_tensor_resized
        # print(pixel_values.max(), pixel_values.min())
        # # np.save('test.npy', image_tensor_new_norm.detach().cpu().numpy())
        # # print(image_tensor_new_norm.max(), image_tensor_new_norm.min(), image_tensor_new_norm.mean(), image_tensor_new_norm.var())
        # gt_idx_Bl = self.vae.img_to_idxBl(image_tensor_new_norm)
        # gt_BL = torch.cat(gt_idx_Bl, dim=1)
        # x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl).detach().clone()
        # B = features.shape[0]
        # V = self.vae.vocab_size
        # dtype = features_vision.dtype

        # if not self.var.is_float32:
        #     self.var = self.var.float()
        #     self.var.is_float32 = True
        # logits_BLV = self.var(features_vision.float(), x_BLCv_wo_first_l.float()).to(dtype)
        # loss = train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
        # image_loss = loss.sum(dim=-1).mean()

        # vae = self.vae
        # noise_scheduler_copy = self.noise_scheduler_copy
        # transformer = self.transformer
        # vae_config_shift_factor = vae.config.shift_factor
        # vae_config_scaling_factor = vae.config.scaling_factor
        # from diffusers.training_utils import (
        #     compute_loss_weighting_for_sd3,
        #     compute_density_for_timestep_sampling,
        # )
        # weighting_scheme = 'logit_normal'
        
        # if random.random() < 0.1:
        #     bs = image_features.shape[0]
        #     prompt_embeds = self.negative_embed.repeat(bs, 1, 1)
        #     # print(prompt_embeds.shape, image_features.shape)
        #     # assert prompt_embeds.shape == image_features
        #     pooled_prompt_embeds = self.negative_embed_pooled.repeat(bs, 1)
        # else:
        #     prompt_embeds = image_features
        #     pooled_prompt_embeds = self.features_proj_vision_pooler(image_features[:, 0])

        # def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        #     sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        #     schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
        #     timesteps = timesteps.to(device)
        #     step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        #     sigma = sigmas[step_indices].flatten()
        #     while len(sigma.shape) < n_dim:
        #         sigma = sigma.unsqueeze(-1)
        #     return sigma
        # with torch.no_grad():
        #     model_input = vae.encode(pixel_values).latent_dist.sample()
        #     model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
        #     noise = torch.randn_like(model_input)
        #     bsz = model_input.shape[0]

        #     u = compute_density_for_timestep_sampling(
        #         weighting_scheme=weighting_scheme,
        #         batch_size=bsz,
        #         logit_mean=0.0,
        #         logit_std=1.0,
        #         mode_scale=1.29,
        #     )
        #     indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        #     timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
        #     sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        #     noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # model_pred = transformer(
        #     hidden_states=noisy_model_input,
        #     timestep=timesteps,
        #     encoder_hidden_states=prompt_embeds,
        #     pooled_projections=pooled_prompt_embeds,
        #     return_dict=False,
        # )[0]
        # if torch.isnan(model_pred).any():
        #     logging.info('model_pred nan', image_features.max(), image_features.min(), text_features.max(), text_features.min())
        #     # raise
        # # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # # Preconditioning of the model outputs.
        # model_pred = model_pred * (-sigmas) + noisy_model_input
        # weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
        # target = model_input
        # image_loss = torch.mean(
        #     (weighting * (model_pred - target) ** 2).reshape(target.shape[0], -1),
        #     1,
        # ).mean()

        # text_loss = torch.tensor(0)
        # Project features to decoder's hidden size
        # text_input_ids = raw_text
        # features_len = text_features.shape[1]
        # raw_text = [item.replace('</s>', '').replace('<s>', '') for item in raw_text]
        # text_input_ids = self.tokenizer.batch_encode_plus(
        #     raw_text, 
        #     return_tensors='pt',
        #     max_length=77,
        #     padding='max_length',
        #     padding_side='right',
        #     truncation=True,
        # ).input_ids.to(device)
        # text_embeddings = self.decoder_model.model.embed_tokens(text_input_ids)  # [batch_size, seq_len_text, hidden_size]
        # # text_embeddings = torch.nan_to_num(text_embeddings, nan=0.0)
        # combined_embeddings = torch.cat([text_features, text_embeddings[:, 1:]], 1)
        # attention_mask = (text_input_ids != self.tokenizer.pad_token_id).to(device).long()
        # attention_mask = torch.cat([torch.ones_like(attention_mask[:, :features_len-1]), attention_mask], 1)
        # transformer_outputs = self.decoder_model(
        #     inputs_embeds=combined_embeddings,
        #     attention_mask=attention_mask,
        #     use_cache=False,
        #     output_attentions=False,
        #     output_hidden_states=False,
        #     return_dict=True,
        # )
        # text_logits = transformer_outputs.logits
        # # text_logits = torch.nan_to_num(text_logits, nan=0.0)
        # # Prepare labels for loss calculation (teacher forcing)
        # shifted_labels = text_input_ids[:, 1:].contiguous()  # [batch_size, seq_len_text - 1]
        # shifted_logits = text_logits[:, features_len-1:-1, :].contiguous()  # [batch_size, seq_len_text - 1, vocab_size]
        # # Compute text prediction loss (CrossEntropyLoss)
        # text_loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # text_loss = text_loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        # text_loss = torch.tensor(0).to(image_loss)
        return recon_loss

class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        # from transformers import AutoModel
        # # load the model and processor
        # ckpt = "google/siglip2-so400m-patch14-224"
        # model = AutoModel.from_pretrained(ckpt).train().to(torch.bfloat16)
        # self.logit_scale = model.logit_scale
        # self.logit_bias = model.logit_bias

        # self.visual = model.vision_model
        # self.visual.image_size = self.visual.config.image_size
        # self.text = model.text_model
        # del model

        self.visual, self.visual_feature_dim = _build_vision_tower(
            embed_dim, vision_cfg, quick_gelu, cast_dtype)
        
        self.text, self.text_feature_dim = _build_text_tower(
            embed_dim, text_cfg, quick_gelu, cast_dtype)

        
        # from peft import LoraConfig, get_peft_model
        # lora_config = LoraConfig(
        #     r=16,  # Rank of the LoRA matrices
        #     lora_alpha=64,  # Scaling factor
        #     target_modules=["q_proj", "k_proj", "v_proj"],  # Modules to apply LoRA
        #     lora_dropout=0.1,
        #     bias="none",
        # )

        # Apply LoRA to the model
        # self.text_encoder = get_peft_model(self.text_encoder, lora_config)
        # lora_config = LoraConfig(
        #     r=16,  # Rank of the LoRA matrices
        #     lora_alpha=64,  # Scaling factor
        #     target_modules=["qkv"],  # Modules to apply LoRA
        #     lora_dropout=0.1,
        #     bias="none",
        # )
        # self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

        # # Set self.visual to evaluation mode and disable gradients
        # self.visual.eval()
        # for param in self.visual.parameters():
        #     param.requires_grad = False

        # # Set self.text to evaluation mode and disable gradients
        # self.text.eval()
        # for param in self.text.parameters():
        #     param.requires_grad = False

        self.context_length = 64
        self.vocab_size = 256000
        
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)

        self.logit_scale_image = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias_image = nn.Parameter(torch.ones([]) * init_logit_bias)

        self.logit_scale_text = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias_text = nn.Parameter(torch.ones([]) * init_logit_bias)

        hidden_size = embed_dim
        intermediate_size = hidden_size * 3
        num_attention_heads = 16

        decoder_hidden_size = 512
        probe_size = 197
        out_probe_size = 1
        # self.pooling_head_toimage = MultiheadAttentionPoolingHead(
        #     hidden_size, 
        #     intermediate_size, 
        #     num_attention_heads, 
        #     output_dim=decoder_hidden_size, 
        #     probe_size=probe_size, 
        #     out_probe_size=out_probe_size, 
        #     num_layers=1,
        # )
        # self.pooling_head_totext = nn.Linear(embed_dim, decoder_hidden_size, bias=False)
        # probe_size_totext = 1
        # out_probe_size_totext = 1
        # self.vision_pooling_head_totext = MultiheadAttentionPoolingHead(
        #     hidden_size, 
        #     intermediate_size, 
        #     num_attention_heads, 
        #     output_dim=decoder_hidden_size, 
        #     probe_size=probe_size_totext,
        #     out_probe_size=out_probe_size_totext, 
        #     num_layers=1,
        # )
        self.image_ct_head = nn.Linear(hidden_size, embed_dim, bias=False)
        self.text_ct_head = nn.Linear(hidden_size, embed_dim, bias=False)
        nn.init.eye_(self.image_ct_head.weight)    
        nn.init.eye_(self.text_ct_head.weight)
        # self.image_ssl_head = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim*4),  # First linear layer
        #     nn.SiLU(),                                     # Activation
        #     nn.Linear(embed_dim*4, embed_dim, bias=False)    # Second linear layer
        # )
        # self.text_ssl_head = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim*4),  # First linear layer
        #     nn.SiLU(),                                     # Activation
        #     nn.Linear(embed_dim*4, embed_dim, bias=False)    # Second linear layer
        # )
        
        # self.text_pool_map = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim*4),  # First linear layer
        #     nn.SiLU(),                                     # Activation
        #     nn.Linear(embed_dim*4, embed_dim, bias=False)    # Second linear layer
        # )
        # self.ct_image_weights = nn.Parameter(torch.ones(1, 128)/128.0)
        # self.ct_text_weights = nn.Parameter(torch.ones(1, 128)/128.0)
        # self.image_norm = nn.LayerNorm(hidden_size)
        # self.image_norm_recon = nn.LayerNorm(hidden_size)
        # self.image_drop = nn.Dropout(0.1)
        # self.image_projection_recon = nn.Linear(hidden_size, embed_dim, bias=True)
        # self.image_projection = nn.Linear(hidden_size, embed_dim, bias=False)
        # self.image_projection_ct = nn.Linear(hidden_size, embed_dim, bias=False)
        
        # hidden_size = self.text.width
        # intermediate_size = hidden_size * 4
        # num_attention_heads = self.text.heads
        self.pad_id = 1
        # self.text_encoder.pad_id
        # self.text_encoder.config
        # self.text_encoder.config.pad_token_id = self.text_encoder.config.eos_token_id
        # self.text_pooling_head_toimage = MultiheadAttentionPoolingHead(hidden_size, intermediate_size, num_attention_heads, output_dim=4096, probe_size=probe_size)
        # self.text_pooling_head_totext = MultiheadAttentionPoolingHead(hidden_size, intermediate_size, num_attention_heads, output_dim=896, probe_size=probe_size)
        # self.text_norm = nn.LayerNorm(hidden_size)
        # self.text_norm_recon = nn.LayerNorm(hidden_size)
        # self.text_drop = nn.Dropout(0.1)
        # self.text_projection_recon = nn.Linear(hidden_size, embed_dim, bias=True)
        # self.text_projection = nn.Linear(hidden_size, embed_dim, bias=False)
        # self.text_projection_ct = nn.Linear(hidden_size, embed_dim, bias=False)

        # self.image_text_decoder_model = ImageTextDecoderModel(features_dim=decoder_hidden_size)
        
        # self.vision_pooling_head_toimage.apply(self._init_weights)
        # self.vision_pooling_head_totext.apply(self._init_weights)

        # self.text_projection_ct.apply(self._init_weights)
        # self.text_projection.apply(self._init_weights)
        # self.text_pooling_head.apply(self._init_weights)
        # self.text_pooling_head_recon.apply(self._init_weights)
        # self.text_norm.apply(self._init_weights)
        # self.text_norm_recon.apply(self._init_weights)

        # self.text_pooling_head_toimage.apply(self._init_weights)
        # self.text_pooling_head_totext.apply(self._init_weights)
        # self.image_projection_ct.apply(self._init_weights)
        # self.image_projection.apply(self._init_weights)
        # self.vision_pooling_head.apply(self._init_weights)
        # self.vision_pooling_head_recon.apply(self._init_weights)
        # self.image_norm.apply(self._init_weights)
        # self.image_norm_recon.apply(self._init_weights)

    def prepare_latent(self, ):
        pass
        # latent_size = 1
        # if self.visual.trunk.attn_pool.latent.shape[1] == 1:
        #     self.visual.trunk.attn_pool.latent_len = latent_size
        #     self.visual.trunk.attn_pool.pool = ''
        #     self.visual.trunk.attn_pool.latent = torch.nn.Parameter(self.visual.trunk.attn_pool.latent.repeat(1, latent_size, 1))
        # latent_size = 4
        # if self.visual.attn_pool_ssl.latent.shape[1] == 1:
        #     self.visual.attn_pool_ssl.latent_len = latent_size
        #     self.visual.attn_pool_ssl.pool = ''
            # self.visual.attn_pool_ssl.latent = torch.nn.Parameter(self.visual.attn_pool_ssl.latent.repeat(1, latent_size, 1))

    def _init_weights(self, module):
        # Initialize weights in a suitable manner
        if isinstance(module, nn.Linear):
            # Xavier initialization for weights
            nn.init.xavier_uniform_(module.weight)
            # Zero initialization for biases
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm is typically initialized with ones for weight and zeros for bias
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups,
                         freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        pass
        # self.visual.set_grad_checkpointing(enable)
        # self.text.set_grad_checkpointing(enable)

    def encode_image(self, image=None, text=None, target_image=None, normalize: bool = False, return_features=False, sample_index=0, average_pool=True, use_ssl_head=False, get_image_text_reconstruction_loss=False):
        # outputs = self.visual(image, use_ssl_head=use_ssl_head)
        # features = self.visual_encoder(image).last_hidden_state
        # if isinstance(outputs, BaseModelOutputWithPooling):
        #     features = outputs.pooler_output
        # else:
        #     features = outputs
        # with torch.no_grad():
        pooler_output, features = self.visual(image)

        # pooled_features = F.normalize(pooled_features, dim=-1)
        # if torch.isnan(features).any():
        #     logging.info('image features nan')
            # raise


        # outputs = self.visual(image)
        # _, pooler_output = outputs[:2]
        
            # output = pooler_output
            
        if get_image_text_reconstruction_loss:
            recon_loss = torch.tensor(0).to(pooler_output)
            # rc_features = self.visual.attn_pool_ssl(features)
            # rc_features = self.visual.fc_norm_ssl(rc_features)
            # rc_features = self.visual.head_drop_ssl(rc_features)
            # rc_features = self.visual.head_ssl(rc_features).unsqueeze(1)
            # if use_ssl_head:
            #     rc_image_features = self.pooling_head_toimage(pooler_output.unsqueeze(1))
            #     rc_text_features = None
            #     raw_text = None
            #     recon_loss = self.image_text_decoder_model.compute_loss(rc_image_features, rc_text_features, target_image, raw_text)

        if use_ssl_head:
            pooler_output = self.image_ct_head(pooler_output)

        if normalize:
            pooler_output = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
            # else:
            #     rc_text_features = self.vision_pooling_head_totext(pooler_output.unsqueeze(1).detach())
            #     raw_text = self.tokenizer.tokenizer.batch_decode(text.detach().cpu().numpy())
            #     rc_image_features = None
                # recon_loss = self.image_text_decoder_model.compute_loss(rc_image_features, rc_text_features, image, raw_text)
            # if torch.isnan(rc_image_features).any():
            #     logging.info('rc_image_features nan')
                # raise
            # if torch.isnan(rc_text_features).any():
            #     logging.info('rc_text_features nan')
                # raise
            # rc_features = torch.nan_to_num(rc_features, nan=0.0)
            # rc_features = self.image_norm_recon(rc_features)
            # rc_features = self.image_projection_recon(rc_features)
            
            # features = self.vision_pooling_head(features)
            # features = self.image_norm(features)
            # features = self.image_drop(features)
            # if use_ssl_head:
            #     features = self.image_projection(features)
            # else:
            #     features = self.image_projection_ct(features)
        
        # ct_features = pooled_features.mean(1)
        # ct_features = pooled_features[:, random.randint(0, pooled_features.shape[1]-1)]
        # ct_features = pooled_features.reshape(-1, ct_features.shape[1] * ct_features.shape[2])
        # if use_ssl_head:
        #     pooled_features = self.image_pool_map(pooled_features)
        # else:
        #     pooled_features = self.text_pool_map(pooled_features)
        # if use_ssl_head:
        #     stride_size = 16
        #     m, n, k = pooled_features.shape
        #     pooled_features = pooled_features.reshape(m, n//16, 16, k).mean(2).flatten(1,2)
        #     # pooled_features = pooled_features.unfold(1, stride_size, stride_size).mean(dim=-1).flatten(1,2)  # Unfold the tensor along n dimension
        #     # pooled_features = pooled_features.mean(1)
        # else:
        #     pooled_features = pooled_features.mean(1)
        # pooled_features = F.normalize(pooled_features, dim=-1)
        # if pooled_features.ndim > 2:
        #     if average_pool:
        #         pooled_features = pooled_features.mean(1)
        #     else:
        #     #     batch_size, seq_len = pooled_features.shape[:2]
        #     #     W = torch.normal(mean=torch.zeros(batch_size, seq_len), std=torch.ones(batch_size, seq_len))
        #     #     W = W / W.sum(dim=1, keepdim=True)
        #     #     W = W.to(pooled_features)
        #         if use_ssl_head:
        #             W = self.ct_image_weights
        #         else:
        #             W = self.ct_text_weights
        #         pooled_features = torch.sum(pooled_features * W.unsqueeze(-1), dim=1)
                # pooled_features = pooled_features[:, sample_index*8:(sample_index+1)*8].flatten(1,2)
        # if use_ssl_head:
        #     pooled_features = pooled_features.flatten(1, 2)
        # else:
        #     pooled_features = pooled_features.mean(1)

        # pooled_features = pooled_features.mean(1)
        # pooled_features = F.normalize(pooled_features, dim=-1)
        
        if return_features:
            return pooler_output, features
        if not get_image_text_reconstruction_loss:
            return pooler_output
        else:
            # raw_text = text
            return pooler_output, recon_loss
        # if use_ssl_head:
        #     if self.visual.attn_pool_ssl is not None:
        #         features = self.visual.attn_pool_ssl(features)
        #     pool_type = self.visual_global_pool
        #     features = global_pool_nlc(features, pool_type=pool_type, num_prefix_tokens=self.visual.num_prefix_tokens)

        #     features = self.visual.fc_norm_ssl(features)
        #     features = self.visual.head_drop_ssl(features)
        #     features = self.visual.head_ssl(features)
        #     # features = self.image_mlp(features)
        # else:
        #     outputs = self.visual.forward_head(outputs)
            # features = self.image_text_mlp(features)
        # if self.image_projection is not None:
        #     features = self.image_projection(features)
        # return F.normalize(features, dim=-1) if normalize else features, loss_image, loss_text

    def encode_text(self, text=None, image=None, normalize: bool = False, use_ssl_head=False, get_image_text_reconstruction_loss=False):
        # print('ori text', text[0].detach().cpu().numpy(), self.pad_id)
        
        # print(self.pad_id)
        # print(text[0].detach().cpu().numpy())
        # attention_mask = (text != self.pad_id).to(text)
        # pooler_output = self.text(text, output_hidden_states=True, attention_mask=attention_mask).hidden_states[-1]
        # if isinstance(outputs, BaseModelOutputWithPooling):
        #     features = outputs.pooler_output
        # else:
        #     features = outputs
        # print(text[0], self.pad_id)
        # attention_mask = (text != self.pad_id).to(text)
        # with torch.no_grad():
        pooler_output, _ = self.text(text)


        # outputs = self.text(text)
        # _, pooler_output = outputs[:2]
        
        if get_image_text_reconstruction_loss:
            recon_loss = torch.tensor(0).to(pooler_output)
            # if use_ssl_head:
            #     rc_text_features = self.pooling_head_totext(pooler_output.unsqueeze(1))
            #     raw_text = self.tokenizer.tokenizer.batch_decode(text.detach().cpu().numpy())
            #     rc_image_features = None
            #     recon_loss = self.image_text_decoder_model.compute_loss(rc_image_features, rc_text_features, image, raw_text)
            
        if use_ssl_head:
            pooler_output = self.text_ct_head(pooler_output)
        if normalize:
            pooler_output = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
                                    #  , attention_mask=attention_mask)
        # pooler_output = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
        # pooler_output = F.normalize(pooler_output, dim=-1)
        # ct_features = pooled_features.unsqueeze(1).repeat(1, 16, 1)
        # ct_features = ct_features.reshape(-1, ct_features.shape[1] * ct_features.shape[2])
        # if torch.isnan(features).any():
        #     logging.info('text features nan')
        # ct_features = pooler_output
        # if get_image_text_reconstruction_loss:
        #     attention_mask = (text != self.pad_id).to(text)
        #     rc_image_features = self.text_pooling_head_toimage(features, attention_mask=attention_mask)
        #     rc_text_features = self.text_pooling_head_totext(features, attention_mask=attention_mask)
        #     if torch.isnan(rc_image_features).any():
        #         logging.info('text rc_image_features nan')
        #         # raise
        #     if torch.isnan(rc_text_features).any():
        #         logging.info('text rc_text_features nan')
            # rc_features = self.text_pooling_head_recon(features, attention_mask=attention_mask)
            # rc_features = self.text_norm_recon(rc_features)
            # rc_features = torch.nan_to_num(rc_features, nan=0.0)
            # rc_features = self.text_projection_recon(rc_features)
            # features = self.text_pooling_head(features, attention_mask=attention_mask)
            # features = self.text_norm(features)
            # features = self.text_drop(features)
            # if use_ssl_head:
            #     features = self.text_projection(features)
            # else:
            #     features = self.text_projection_ct(features)

        # return pooler_output
        #     # features = self.text_mlp(features)
        #     if self.attn_pool_image is not None:
        #         features = self.attn_pool_image(features)
        #     pool_type = self.visual_global_pool
        #     features = global_pool_nlc(features, pool_type=pool_type, num_prefix_tokens=self.visual.num_prefix_tokens)

        #     features = self.fc_norm_image(features)
        #     features = self.head_drop_image(features)
        #     features = self.head_image(features)
        # else:
        #     features = self.image_text_mlp(features)
        # if self.text_projection is not None:
        #     features = self.text_projection(features)
        if not get_image_text_reconstruction_loss:
            return pooler_output
        else:
            return pooler_output, recon_loss

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(
            image, normalize=True) if image is not None else None
        text_features = self.encode_text(
            text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + \
            1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(
        k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    # OpenAI state dicts are partially converted to float16
    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones(
        (batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros(
        (batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    # FIXME detect different token configs (ie no class token, or more)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:
                                                 extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s',
                 old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(
        1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(
        1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_text_pos_embed(state_dict, model, interpolation: str = 'linear', antialias: bool = False):
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, 'positional_embedding', None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, 'positional_embedding', None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, 'text pos_embed width changed!'
    if old_num_pos == num_pos:
        return

    logging.info(
        'Resizing text position embedding num_pos from %s to %s', old_num_pos, num_pos)
    old_pos_embed = old_pos_embed.reshape(
        1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict['positional_embedding'] = new_pos_embed


def get_model_preprocess_cfg(model):
    module = getattr(model, 'visual', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'image_size')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', None)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', None)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, 'visual', model)
    # legacy attribute, keeping for bwd compat
    module.image_mean = preprocess_cfg['mean']
    # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg['std']
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)


def get_model_tokenize_cfg(model):
    module = getattr(model, 'text', model)
    cfg = {}
    context_length = getattr(module, 'context_length', None)
    if context_length is not None:
        cfg['context_length'] = context_length
    vocab_size = getattr(module, 'vocab_size', None)
    if vocab_size is not None:
        cfg['vocab_size'] = vocab_size
    return cfg
