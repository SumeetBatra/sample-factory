import torch
from torch import nn

from algorithms.appo.model_utils import get_obs_shape, nonlinearity, create_standard_encoder, EncoderBase, \
    register_custom_encoder
from algorithms.utils.pytorch_utils import calc_num_elements
from utils.utils import log



class QuadMultiEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing, self_obs_dim=18, neighbor_obs_dim=6, neighbor_hidden_size=32):
        super().__init__(cfg, timing)
        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size

        obs_shape = get_obs_shape(obs_space)
        fc_encoder_layer = cfg.hidden_size
        self.self_encoder = nn.Sequential(
            nn.Linear(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            nn.Linear(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )

        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.neighbor_obs_dim, self.neighbor_hidden_size),
            nonlinearity(cfg),
        )
        self.self_encoder_out_size = calc_num_elements(self.self_encoder, (self.self_obs_dim,))
        self.neighbor_encoder_out_size = calc_num_elements(self.neighbor_encoder, (self.neighbor_obs_dim,))

        self.init_fc_blocks(self.self_encoder_out_size)
        self.init_fc_blocks(self.neighbor_encoder_out_size)



    def forward(self, obs_dict):
        t = torch.randn_like(self.obs_shape)
        x = self.encoder(t)
        return x


def register_models():
    register_custom_encoder('quad_multi_encoder', QuadMultiEncoder)
