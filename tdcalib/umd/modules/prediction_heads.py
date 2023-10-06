import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from umd.modules.position_embeddings import get_2d_sincos_pos_embed
from umd.modules.vision_encoders.clip_model import Transformer, LayerNorm

# class Projection(nn.Module):
#     def __init__(self, d_out: int, p: float=0.5) -> None:
#         super().__init__()
#         self.linear1 = nn.Linear(d_in, d_out, bias=False)
#         self.linear2 = nn.Linear(d_out, d_out, bias=False)
#         self.layer_norm = nn.LayerNorm(d_out)
#         self.drop = nn.Dropout(p)

#     def forward(self, x: torch.Tensor):
#         embed1 = self.linear1(x)
#         embed2 = self.drop(self.linear2(F.gelu(embed1)))
#         embeds = self.layer_norm(embed1 + embed2)
#         return embeds

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # torch.Size([90, 64, 768]) language
        # torch.Size([90, 82, 768]) vision

        # first_token: torch.Size([90, 768])
        # pooled_output1: torch.Size([90, 768])
        # pooled_output2: torch.Size([90, 768])
        
        # method 1:
        # first_token_tensor = hidden_states[:, 0]
        # pooled_output = self.dense(first_token_tensor)
        # pooled_output = self.activation(pooled_output)

        # mean_token_tensor: torch.Size([90, 768])
        # pooled_output1: torch.Size([90, 768])
        # pooled_output2: torch.Size([90, 768])
        
        # method 2:
        # 2.1 calculate average embeddings
        avg_token_tensor = torch.mean(hidden_states, dim = 1) # avg_embeddings shape: (90, 768)

        # 2.2 calculate pooled embeddings
        # pooled_token_tensor = torch.max(hidden_states, dim=1)[0] # pooled_embeddings shape: (90, 768)

        pooled_output = self.dense(avg_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MIMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.patch_size = config["patch_size"]
        self.num_patches = (config["image_size"] // config["patch_size"]) ** 2
        self.decoder_hidden_size = config["mim_decoder_hidden_size"]
        self.decoder_num_layers = config["mim_decoder_num_layers"]
        self.decoder_num_heads = config["mim_decoder_num_heads"]
        self.decoder_num_channels = 3 * config["patch_size"] ** 2

        self.decoder_embed = nn.Linear(self.hidden_size, self.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1,
                                                          self.decoder_hidden_size), requires_grad=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_hidden_size, int(self.num_patches ** .5), True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.decoder = Transformer(self.decoder_hidden_size, self.decoder_num_layers + 1, self.decoder_num_heads)
        self.decoder_norm = LayerNorm(self.decoder_hidden_size)
        self.decoder_pred = nn.Linear(self.decoder_hidden_size, self.patch_size ** 2 * 3, bias=True)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed.to(x.dtype)

        # apply Transformer blocks
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.decoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x
