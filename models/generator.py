import transformers
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from models.config import GeneratorConfig
from typing import Optional, Literal

class GeneratorPreTrainedModel(PreTrainedModel):
    
    config_class = GeneratorConfig
    base_model_prefix = "gan"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
                
class Generator(GeneratorPreTrainedModel):
    
    def __init__(self, config:GeneratorConfig):
        super().__init__(config)
        img_shape = (config.img_size ** 2) * config.img_channels
        self.encoder = GanEncoder(config)
        self.projection = nn.Linear(128 * 2 ** config.num_layer, img_shape)
        self.act = nn.Tanh()
        
        self.post_init()
        
    def forward(self, input: Optional[torch.Tensor]):
        embeddings = self.encoder(input)
        image = self.projection(embeddings)
        
        return self.act(image)
        
class GanEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head = GanLayer(config.latent_dim, 128, config.activation, normalize=False)
        self.layer = nn.ModuleList([GanLayer(128 * 2 ** i, 128 * 2 ** (i+1), config.activation) for i in range(config.num_layer)])
        
    def forward(self, input: Optional[torch.Tensor]):
        output = self.head(input)
        for i, layer_module in enumerate(self.layer):
            output = layer_module(output)
        return output
        
class GanLayer(nn.Module):
    
    def __init__(self, in_feat: int, out_feat: int, activation: str, normalize: bool=True):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.activation = nn.functional.gelu if activation == "gelu" else (nn.ReLU if activation == "relu" else nn.LeakyReLU)
        self.layer = nn.Linear(in_feat, out_feat)
        self.norm = None
        if normalize:
            self.norm = nn.LayerNorm(out_feat, eps=1e-12)
            
    def forward(self, input: Optional[torch.Tensor]):
        output = self.layer(input)
        if self.norm:
            output = self.norm(output)
        return self.activation(output)
    