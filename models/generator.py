import transformers
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from models.config import GeneratorConfig
from typing import Optional, Literal
from utils.train_utils import kaiming_normal_
class GeneratorPreTrainedModel(PreTrainedModel):
    
    config_class = GeneratorConfig
    base_model_prefix = "gan"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            # kaiming_normal_(module.weight, mode="fan_out", nonlinearity="gelu")
            nn.init.normal_(module.weight, 0.0, 0.02)
            
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)
            
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
                
class Generator(GeneratorPreTrainedModel):
    
    def __init__(self, config:GeneratorConfig):
        super().__init__(config)
        self.config = config
        self.encoder = GanEncoder(config)
        self.projection = nn.Conv2d(64, config.img_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.Tanh()
        
        self.post_init()
        
    def forward(self, input: Optional[torch.Tensor]):
        embeddings = self.encoder(input)
        image = self.projection(embeddings)
        image = self.act(image)
        
        return image


class GanEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_size = config.img_size // (2 ** config.num_layer)
        self.init_dim = (64 * (2 ** config.num_layer))
        self.head = nn.Linear(config.latent_dim + config.noise_dim, self.init_dim * (self.init_size ** 2))
        self.head_norm = nn.BatchNorm2d(self.init_dim)
        self.layer = nn.ModuleList([GanLayer(in_channels=64 * (2 ** (config.num_layer - i)), 
                                             out_channels=64 * (2 ** (config.num_layer - (i+1))), 
                                             activation = config.activation) for i in range(config.num_layer)])

    def forward(self, input: Optional[torch.Tensor]):
        noise = torch.randn(input.size(0), self.config.noise_dim).to(input.device)
        output = self.head(torch.cat((input, noise), dim=1))
        output = output.view(output.shape[0], self.init_dim,
                             self.init_size, self.init_size)
        output = self.head_norm(output)
        for i, layer_module in enumerate(self.layer):
            output = layer_module(output)
        return output
        
class GanLayer(nn.Module):
    
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu", padding: int = 1, factor: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = nn.functional.gelu if activation == "gelu" else (nn.functional.relu if activation == "relu" else nn.functional.leaky_relu)
        self.block = self.conv_block(nn.Conv2d, in_channels, out_channels, kernel_size, stride, padding, factor)
    def conv_block(self, func, in_channels, out_channels, kernel_size, stride, padding, factor):
        return nn.Sequential(
            nn.Upsample(scale_factor=factor),
            func(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            func(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )
            
    def forward(self, input: Optional[torch.Tensor]):
        _x = self.block(input)
        return self.activation(_x)