import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .MultiHeadAttention import DeformableHeadAttention
import copy
class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, C, M, K, n_levels, last_feat_height, last_feat_width, d_ffn=1024, dropout=0.1, normalize_before=False):
        super().__init__()
        """
        Args:
            - C: Number of expected features in the decoder inputs.
            - d_ffn : feed forward network dimension.
            - n_levels: multiscale parameter.
            - M: number of attention heads.
            - K: number of sampling points.
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
        # the feed forward network
        self.ffn = FeedForward(C, d_ffn)
        
    def forward(self, query_objects, out_encoder, ref_points, tgt_mask = None, memory_masks = None,
                tgt_key_padding_mask = None, memory_key_padding_masks = None,positional_embeddings = None,
                query_poses = None):
        """
        Args:
            - query_objects : query_embedding passed to the transformer.
            - out_encoder : result of the encoder.
            - ref_points : linear projection of tgt to 2 dim (in the encoder).
            - memory_key_padding_masks : the mask passed to the transformer.
            - tgt_key_padding_mask : the mask for target keys per batch.
            - positional_embeddings : positional embeddings passed to the transformer.
            - query_poses : query_embed passed to the transformer.
        """
        if self.normalize_before:
            return self.forward_pre_norm(query_objects, out_encoder,ref_points,tgt_mask,
                             memory_masks,tgt_key_padding_mask,
                             memory_key_padding_masks, positional_embeddings, query_poses)
        return self.forward_post_norm(query_objects, out_encoder,ref_points,tgt_mask,
                         memory_masks,tgt_key_padding_mask,
                         memory_key_padding_masks, positional_embeddings, query_poses)

    def forward_post_norm(self, query_objects, out_encoder, ref_points, tgt_mask = None, memory_masks = None, 
                          tgt_key_padding_mask = None, memory_key_padding_masks = None, positional_embeddings = None, 
                          query_poses = None):
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
        query_objects_2, attention_weights = self.def_attn(query_objects, out_encoder, ref_points,query_mask=None, x_masks=memory_key_padding_masks)
        query_objects = query_objects + self.dropout2(query_objects_2)
        query_objects = self.norm2(query_objects)
        query_objects = self.ffn(query_objects)
        query_objects = self.norm3(query_objects) #post normalization
        # B, L, 1, C -> L, B, C
        query_objects = query_objects.squeeze(dim=2)
        query_objects = query_objects.transpose(0, 1).contiguous()
        return query_objects

    def forward_pre_norm(self, query_objects, out_encoder, ref_points, tgt_mask = None, memory_masks = None, 
                         tgt_key_padding_mask = None, memory_key_padding_masks = None, positional_embeddings = None,
                         query_poses = None):

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
        query_objects = query_objects + self.dropout2(query_objects_2)
        query_objects = self.norm3(query_objects)
        query_objects = self.ffn(query_objects)
        # B, L, 1, C -> L, B, C
        query_objects = query_objects.squeeze(dim=2)
        query_objects = query_objects.transpose(0, 1).contiguous()

        return query_objects


class DeformableTransformerDecoder(nn.Module):
    
    def __init__(self, decoder_layer, num_layers, norm=None,return_intermediate=False):
        """
        Args:
            - decoder_layer: an instance of the DeformableTransformerDecoderLayer() class.
            - num_layers: the number of sub-decoder-layers in the decoder.
            - norm: the layer normalization component (optional).
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = norm
        
    def forward(self, query_objects, out_encoder, ref_point, tgt_mask = None, memory_masks = None,
                tgt_key_padding_mask = None, memory_key_padding_masks = None, positional_embeddings = None, query_pos = None):
        
        # input of the decoder layers
        output = query_objects
        intermediate = []
        for layer in self.layers:
            output = layer(output, out_encoder, ref_point, tgt_mask = tgt_mask,memory_masks = memory_masks,tgt_key_padding_mask = tgt_key_padding_mask,
                           memory_key_padding_masks = memory_key_padding_masks,positional_embeddings= positional_embeddings,query_poses = query_pos)
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
    """Simple Feed Forward Network"""
    def __init__(self, C=256, d_ffn=1024, dropout=0.5):
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
