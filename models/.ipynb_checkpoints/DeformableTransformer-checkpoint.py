# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .MultiheadAttention import DeformableHeadAttention, generate_ref_points


class Transformer(nn.Module):

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=False,
                 scales=4,
                 k=4,
                 last_height=16,
                 last_width=16, ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(C=d_model,
                                                M=nhead,
                                                K=k,
                                                n_levels=scales,
                                                last_feat_height=last_height,
                                                last_feat_width=last_width,
                                                d_ffn=dim_feedforward,
                                                dropout=dropout,
                                                normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(C=d_model,
                                                M=nhead,
                                                K=k,
                                                n_levels=scales,
                                                last_feat_height=last_height,
                                                last_feat_width=last_width,
                                                d_ffn=dim_feedforward,
                                                dropout=dropout,
                                                normalize_before=normalize_before)
        
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.query_ref_point_proj = nn.Linear(d_model, 2)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: List[Tensor],
                masks: List[Tensor],
                query_embed,
                pos_embeds: List[Tensor]):
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
        memory = self.encoder(src,
                              ref_points,
                              padding_mask=masks,
                              pos_encodings=pos_embeds)

        # L, B, C
        query_ref_point = self.query_ref_point_proj(tgt)
        query_ref_point = F.sigmoid(query_ref_point)

        # Decoder Layers, L, B ,C
        hs = self.decoder(tgt, memory,
                          query_ref_point,
                          memory_key_padding_masks=masks,
                          positional_embeddings=pos_embeds,
                          query_pos=query_embed)

        return hs, query_ref_point, memory



class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, input_features, ref_points, input_masks=None, pos_encodings=None, padding_mask=None):
        output = input_features
        for layer in self.layers:
            outputs = layer(output, ref_points, input_masks =input_masks, padding_masks=padding_mask, pos_encodings =pos_encodings)
        if self.norm is not None:
            for i, output in enumerate(outputs):
                outputs[i] = self.norm(output)
        return outputs

# Decoder 
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None,return_intermediate=False):
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = norm
        

    def forward(self, query_objects, out_encoder,
                ref_point,
                tgt_mask = None,
                memory_masks = None,
                tgt_key_padding_mask = None,
                memory_key_padding_masks = None,
                positional_embeddings = None,
                query_pos = None):
        
        
        

        # input of the decoder layers
        output = query_objects
        intermediate = []
        for layer in self.layers:
            output = layer(  output, out_encoder,
                             ref_point,
                             tgt_mask = tgt_mask,
                             memory_masks = memory_masks,
                             tgt_key_padding_mask = tgt_key_padding_mask,
                             memory_key_padding_masks = memory_key_padding_masks,
                             positional_embeddings= positional_embeddings,
                             query_poses = query_pos)
            
            
            
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


    


class FeedForward(nn.Module):
    def __init__(self, C=256, d_ffn=1024, dropout=0.1):
        super(FeedForward, self).__init__()
        self.C = C
        self.d_ffn = d_ffn
        self.linear1 = nn.Linear(C, d_ffn)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, C)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, attended):
        attended_tmp = self.linear2(self.dropout1(F.relu(self.linear1(attended))))
        attended = attended + self.dropout2(attended_tmp)
        return attended 
class TransformerEncoderLayer(nn.Module):
    def __init__(self,C, M, K, n_levels, last_feat_height, last_feat_width, need_attn=False, d_ffn=2048,
                 dropout=0.1, normalize_before=False):
        super().__init__()
        """
        Args:
            - C: dimesension of the embeddings
            - d_ffn : feed forward network dim
            - n_levels: multiscale parameter
            - M: number of attention heads
            - K: number of sampling points
        """
        # self attention
        
        
        
        
        self.self_attn = DeformableHeadAttention(last_height = last_feat_height,last_width = last_feat_width, C = C, M=M, K=K, L = n_levels, dropout=dropout, return_attentions = False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.norm3 = nn.LayerNorm(C)
        self.normalize_before = normalize_before
        # g_theta
        self.ffn = FeedForward(C, d_ffn, dropout)

    def forward_pre_norm(self, input_features,
                ref_points,
                input_masks=None,
                padding_masks=None,
                pos_encodings=None):
        if input_masks is None:
            input_masks = [None] * len(input_features)

        if padding_masks is None:
            padding_masks = [None] * len(input_features)

        if pos_encodings is None:
            pos_encodings = [None] * len(pos_encodings)
        feats = []
        features = [feature + pos for (feature, pos) in zip(input_features, pos_encodings)] # add pos encodings to features
        for q, ref_point, key_padding_mask, pos in zip(features, ref_points, padding_masks, pos_encodings):
            feat = self.norm1(q) # pre normalization
            feat, attention = self.self_attn(feat, features, ref_point, key_padding_mask, padding_masks)
            q = q + self.dropout1(feat) #
            q = self.norm2(q) #
            q = self.ffn(q)
            feats.append(q)

        return feats
    def forward_post_norm(self, input_features,
                ref_points,
                input_masks=None,
                padding_masks=None,
                pos_encodings=None):
        if input_masks is None:
            input_masks = [None] * len(input_features)

        if padding_masks is None:
            padding_masks = [None] * len(input_features)

        if pos_encodings is None:
            pos_encodings = [None] * len(pos_encodings)
        feats = []
        features = [feature + pos for (feature, pos) in zip(input_features, pos_encodings)] # add pos encodings to features
        for q, ref_point, key_padding_mask, pos in zip(features, ref_points, padding_masks, pos_encodings):
            feat, attention = self.self_attn(q, features, ref_point, key_padding_mask, padding_masks)
            q = q + self.dropout1(feat) #
            q = self.norm1(q) #
            q = self.ffn(q)
            q = self.norm2(q) # post normalization
            feats.append(q)
        return feats

    def forward(self, input_features,
                ref_points,
                input_masks = None,
                padding_masks= None,
               pos_encodings = None):
        if self.normalize_before:
            return self.forward_pre_norm(input_features, ref_points, input_masks, padding_masks, pos_encodings)
        return self.forward_post_norm(input_features, ref_points, input_masks, padding_masks, pos_encodings)
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, C,
                 M,
                 K,
                 n_levels,
                 last_feat_height,
                 last_feat_width,
                 d_ffn=1024,
                 dropout=0.1,
                 normalize_before=False):
        super().__init__()
        """
        Args:
            - C: dimesension of the embeddings
            - d_ffn : feed forward network dim
            - n_levels: multiscale parameter
            - M: number of attention heads
            - K: number of sampling points
        """

        # Deformable Attention part
        self.def_attn = DeformableHeadAttention(last_height = last_feat_height,last_width = last_feat_width, C = C, M=M, K=K, L = n_levels, dropout=dropout, return_attentions = False) 
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.norm3 = nn.LayerNorm(C)
        # Proper Attention Part
        self.self_attn = nn.MultiheadAttention(C, M, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.normalize_before = normalize_before
        # the feed forwrd network
        self.ffn = FeedForward(C, d_ffn)
    def forward(self, query_objects, out_encoder,
                     ref_points,
                     tgt_mask = None,
                     memory_masks = None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_masks = None,
                     positional_embeddings = None,
                     query_poses = None):
        if self.normalize_before:
            return self.forward_pre_norm(query_objects, out_encoder,ref_points,tgt_mask,
                             memory_masks,tgt_key_padding_mask,
                             memory_key_padding_masks, positional_embeddings, query_poses)
        return self.forward_post_norm(query_objects, out_encoder,ref_points,tgt_mask,
                         memory_masks,tgt_key_padding_mask,
                         memory_key_padding_masks, positional_embeddings, query_poses)

    def forward_post_norm(self, query_objects, out_encoder,
                     ref_points,
                     tgt_mask = None,
                     memory_masks = None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_masks = None,
                     positional_embeddings = None,
                     query_poses = None):

        """
        Args:
            - tgt : query_embedding passed to the transformer
            - memory : result of the encoder
            - ref_point : linear projection of tgt to 2 dim (in the encoder)
            - memory_key_padding_masks : mask passed to the transformer
            - poses : positional embeddings passed to the transformer
            - query_pos : query_embed passed to the transformer
        """

        # self attention
        q = query_objects + query_poses
        k = q
        query_objects_2 = self.self_attn(q, k, value=query_objects, attn_mask=tgt_mask,key_padding_mask=tgt_key_padding_mask)[0]
        query_objects = query_objects + self.dropout2(query_objects_2)
        query_objects = self.norm1(query_objects)
        # get the output of the encoder with positional embeddings
        out_encoder = [ tensor + pos for tensor, pos in zip(out_encoder, positional_embeddings)] #?
        #query_objects is of same shape as nn.Embedding(number of object queries, C)

        # L, B, C -> B, L, 1, C | L: number of object queries, B: size of batch
        query_objects = query_objects.transpose(0, 1).unsqueeze(dim=2)
        ref_points = ref_points.transpose(0, 1).unsqueeze(dim=2)

        # B, L, 1, 2
        query_objects_2, attention_weights = self.def_attn(query_objects, out_encoder, ref_points,query_mask=None,                                     x_masks=memory_key_padding_masks)
        """if self.need_attn:
            self.attns.append(attns)"""

        query_objects = query_objects + self.dropout2(query_objects_2)
        query_objects = self.norm2(query_objects)
        query_objects = self.ffn(query_objects)
        query_objects = self.norm3(query_objects) #post normalization
        # B, L, 1, C -> L, B, C
        query_objects = query_objects.squeeze(dim=2)
        query_objects = query_objects.transpose(0, 1).contiguous()

        return query_objects

    def forward_pre_norm(self, query_objects, out_encoder,
                     ref_points,
                     tgt_mask = None,
                     memory_masks = None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_masks = None,
                     positional_embeddings = None,
                     query_poses = None):

        """
        Args:
            - query_objects : query_embedding passed to the transformer
            - out_encoder : result of the encoder
            - ref_points : linear projection of tgt to 2 dim (in the encoder)
            - memory_key_padding_masks : mask passed to the transformer
            - positional_embeddings : positional embeddings passed to the transformer
            - query_poses : query_embed passed to the transformer
        """

        # self attention
        query_objects_2 = self.norm1(query_objects)
        q = query_objects_2 + query_poses
        k = q
        query_objects_2 = self.self_attn(q, k, value=query_objects, attn_mask=tgt_mask,key_padding_mask=tgt_key_padding_mask)[0]
        query_objects = query_objects + self.dropout2(query_objects_2)
        query_objects_2 = self.norm2(query_objects)
        # get the output of the encoder with positional embeddings
        out_encoder = [ tensor + pos for tensor, pos in zip(out_encoder, positional_embeddings)]
        #query_objects is of same shape as nn.Embedding(number of object queries, C)

        # L, B, C -> B, L, 1, C | L: number of object queries, B: size of batch
        query_objects = query_objects.transpose(0, 1).unsqueeze(dim=2)
        query_ref_point = query_ref_point.transpose(0, 1).unsqueeze(dim=2)

        # B, L, 1, 2
        query_objects_2, attention_weights = self.def_attn(q, out_encoder, query_ref_point, query_mask=None, x_masks=memory_key_padding_masks)
        """if self.need_attn: 
            self.attns.append(attns)"""

        query_objects = query_objects + self.dropout2(query_objects_2)
        query_objects = self.norm3(query_objects)
        query_objects = self.ffn(query_objects)

        # B, L, 1, C -> L, B, C
        query_objects = query_objects.squeeze(dim=2)
        query_objects = query_objects.transpose(0, 1).contiguous()

        return query_objects

def _get_clones(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        scales=args.scales,
        k=args.k,
        last_height=args.last_height,
        last_width=args.last_width
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
