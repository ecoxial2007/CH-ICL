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

        self.mhca_i2t = ResidualCrossAttentionBlock(config.d_input, config.n_ca_heads, config.ca_dropout, None, 'mh')
        self.mhca_t2i = ResidualCrossAttentionBlock(config.d_input, config.n_ca_heads, config.ca_dropout, None, 'mh')


        self.method = config.method
        if config.method == 'pubmed':
            self.linear_merge_image = nn.Linear(768, 512)
            self.linear_merge = nn.Linear(config.d_input * 2, 512)
            self.mhca_m2a = ResidualCrossAttentionBlock(512, 8, config.ca_dropout, None, 'mh')

        elif config.method == 'pmcmed':
            self.linear_merge_image = nn.Linear(2048, 768)
            self.linear_merge = nn.Linear(config.d_input * 2, 768)
            self.mhca_m2a = ResidualCrossAttentionBlock(768, 12, config.ca_dropout, None, 'mh')

        else:
            self.linear_merge_image = nn.Linear(config.d_input, 512)
            self.linear_merge = nn.Linear(config.d_input * 2, 512)
            self.mhca_m2a = ResidualCrossAttentionBlock(512, 8, config.ca_dropout, None, 'mh')

        self.linear_merge_query = nn.Linear(768, 15)

    def forward(self, item_dict):
        if self.method == 'pmcmed':
            # bFeature = rearrange(item_dict['patch_features'].squeeze(dim=1), 'b c t -> t b c')
            bFeature = rearrange(item_dict['image_features'], 'b t c -> t b c')
            tFeature = rearrange(item_dict['text_query_features'].unsqueeze(dim=1), 'b n c -> n b c')
        else:
            bFeature = rearrange(item_dict['patch_features'].squeeze(dim=1), 'b t c -> t b c')
            tFeature = rearrange(item_dict['text_query_token_features'], 'b n c -> n b c')

        aFeature = rearrange(item_dict['text_cands_features'], 'b t c -> t b c')

        # t, b, c = bFeature.shape
        # bFeature = bFeature.reshape(-1, c)
        # bFeature = self.linear_merge_image(bFeature)
        # if self.method == 'pubmed':
        #     bFeature = bFeature.reshape(t, b, 512)
        # else:
        #     bFeature = bFeature.reshape(t, b, 768)

        bFeature_merge = self.mhca_i2t(bFeature, tFeature, tFeature).mean(dim=0)
        tFeature_merge = self.mhca_t2i(tFeature, bFeature, bFeature).mean(dim=0)

        Feature_merge = torch.cat([bFeature_merge, tFeature_merge], dim=-1)
        Feature_merge = self.linear_merge(Feature_merge).unsqueeze(dim=0)
        Feature_merge = self.mhca_m2a(Feature_merge, aFeature, aFeature).mean(dim=0)
        return Feature_merge, None, None



