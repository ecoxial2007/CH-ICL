from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat
from utils.attn import ResidualCrossAttentionBlock
from open_clip.model import CustomTextCLIP

import json

class LGVAConfig:
    cast_dtype: str = 'fp16'
    config_path: str = 'src/backbone/BiomedCLIP/open_clip_config.json'

    def __init__(self, n_ca_heads, ca_dropout, d_input, method):
        self.n_ca_heads = n_ca_heads
        self.ca_dropout = ca_dropout
        self.d_input = d_input
        self.method = method

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.pretrained_cfg = config['preprocess_cfg']
        self.model_cfg = config['model_cfg']

    @classmethod
    def from_args(cls, args):
        return cls(n_ca_heads=args.n_ca_heads,
                   ca_dropout=args.ca_dropout,
                   d_input=args.d_input,
                   method=args.method,
                   )



class LGVAModel(nn.Module):

    def __init__(self, config: LGVAConfig, device):
        super().__init__()
        self.config = config
        self.device = device
        self.backbone = CustomTextCLIP(**config.model_cfg, cast_dtype=config.cast_dtype)
        checkpoint = torch.load('./src/backbone/BiomedCLIP/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.pth')
        self.backbone.load_state_dict(checkpoint, strict=False)

        self.mhca_i2t = ResidualCrossAttentionBlock(config.d_input, config.n_ca_heads, config.ca_dropout, None, 'mh')
        self.mhca_t2i = ResidualCrossAttentionBlock(config.d_input, config.n_ca_heads, config.ca_dropout, None, 'mh')
        self.mhca_m2a = ResidualCrossAttentionBlock(512, 8, config.ca_dropout, None, 'mh')

        self.linear_merge = nn.Linear(config.d_input * 2, 512)
        self.linear_merge_image = nn.Linear(config.d_input, 512)
        self.linear_merge_query = nn.Linear(768, 15)

        self.classifier = nn.Linear(512, 16555)

    def forward(self, item_dict):
        bFeature = item_dict['patch_features'].squeeze(dim=1)
        bFeature = rearrange(bFeature, 'b t c -> t b c')
        aFeature = rearrange(item_dict['text_cands_features'], 'b t c -> t b c')
        bFeature = self.linear_merge_image(bFeature)
        Feature_merge = self.mhca_m2a(bFeature, aFeature, aFeature).mean(dim=0)
        return self.classifier(Feature_merge), None, None



