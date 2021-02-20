import copy
from typing import Optional, List
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from .encoder import DeformableTransformerEncoderLayer, DeformableTransformerEncoder
from .decoder import DeformableTransformerDecoderLayer, DeformableTransformerDecoder
from .MultiHeadAttention import DeformableHeadAttention, generate_ref_points


class DeformableTransformer(nn.Module):
    """Transformer module with deformable attention"""
    def __init__(self,
                 d_model=512,nhead=8, num_encoder_layers=6,num_decoder_layers=6, dim_feedforward=2048,dropout=0.1,
                 normalize_before=False, return_intermediate_dec=False, scales=4,k=4, last_height=16, last_width=16):
        """
        Args:
            - d_model : number of expected features in the encoder and decoder inputs.
            - nhead : number of heads.
            - num_encoder_layers : number of encoder layers.
            - num_decoder_layers : number of decoder layers.
            - dim_feedforward : feed forward network dimension.
            - dropout : the dropout value.
            - normalize_before : True if normalization is to be used before computing attention scores.
            - return_intermediate_dec : True if auxiliary decoding losses are to be used.
            - scales : multi-scale parameter.
            - k : number of sampling points.
            - last_height : smallest feature height.
            - last_width:smallest feature width.
        """
        super().__init__()

        encoder_layer = DeformableTransformerEncoderLayer(C=d_model, M=nhead, K=k, n_levels=scales, last_feat_height=last_height, 
                                                          last_feat_width=last_width,d_ffn=dim_feedforward,dropout=dropout, 
                                                          normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = DeformableTransformerDecoderLayer(C=d_model, M=nhead, K=k, n_levels=scales, last_feat_height=last_height,
                                                          last_feat_width=last_width,d_ffn=dim_feedforward, dropout=dropout,
                                                        normalize_before=normalize_before)
        
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.C = d_model
        self.nhead = nhead

        self.query_ref_point_proj = nn.Linear(d_model, 2)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, masks, query_embed, pos_embeds):
        """
        Args:
            src : batched images.
            masks : masks for input images.
            query_embed: query embeddings (objects Deformable DETR can detect in an image).
            pos_embeds: inputs positional embeddings.
        """
        bs = src[0].size(0)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # B, C H, W -> B, H, W, C
        for index in range(len(src)):
            src[index] = src[index].permute(0, 2, 3, 1)
            pos_embeds[index] = pos_embeds[index].permute(0, 2, 3, 1)

        # B, H, W, C
        ref_points = []
        for tensor in src:
            _, height, width, _ = tensor.shape
            ref_point = generate_ref_points(width=width,
                                            height=height)
            ref_point = ref_point.type_as(src[0])
            # H, W, 2 -> B, H, W, 2
            ref_point = ref_point.unsqueeze(0).repeat(bs, 1, 1, 1)
            ref_points.append(ref_point)

        tgt = torch.zeros_like(query_embed)

        # List[B, H, W, C]
        memory = self.encoder(src, ref_points, padding_mask=masks, pos_encodings=pos_embeds)

        # L, B, C
        query_ref_point = self.query_ref_point_proj(tgt)
        query_ref_point = F.sigmoid(query_ref_point)

        # Decoder Layers, L, B ,C
        hs = self.decoder(tgt, memory, query_ref_point, memory_key_padding_masks=masks, positional_embeddings=pos_embeds, query_pos=query_embed)

        return hs, query_ref_point, memory


