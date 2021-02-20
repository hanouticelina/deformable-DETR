import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .MultiHeadAttention import DeformableHeadAttention
import copy
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,C, M, K, n_levels, last_feat_height, last_feat_width, d_ffn=2048,
                 dropout=0.1, normalize_before=False):
        super().__init__()
        """
        Args:
            - C: Number of expected features in the encoder inputs.
            - M: number of attention heads.
            - K: number of sampling points.
            - n_levels: multiscale parameter.
            - last_feat_height : smallest feature height.
            - last_feat_width : smallest feature width.
            - d_ffn : feed forward network dimension.
            

        """
        # self attention
        self.self_attn = DeformableHeadAttention(last_height = last_feat_height,last_width = last_feat_width, C = C, M=M, K=K, L = n_levels, dropout=dropout, return_attentions = False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.norm3 = nn.LayerNorm(C)
        self.normalize_before = normalize_before
        self.ffn = FeedForward(C, d_ffn, dropout)
        
    def forward(self, input_features, ref_points, input_masks = None, padding_masks= None, pos_encodings = None):
        """
        Args:
            - input_features : the sequence to the encoder.
            - ref_points : reference points.
            - input_masks : the mask for the input keys.
            - padding_masks : masks for padded inputs.
            - pos_embeddings : positional embeddings passed to the transformer.
            
        """
        if self.normalize_before:
            return self.forward_pre_norm(input_features, ref_points, input_masks, padding_masks, pos_encodings)
        return self.forward_post_norm(input_features, ref_points, input_masks, padding_masks, pos_encodings)
    
    def forward_pre_norm(self, input_features, ref_points, input_masks=None, padding_masks=None, pos_encodings=None):
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
    def forward_post_norm(self, input_features, ref_points, input_masks=None, padding_masks=None,
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




class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        Args:
            - decoder_layer: an instance of the DeformableTransformerEncoderLayer() class.
            - num_layers: the number of sub-decoder-layers in the decoder.
            - norm: the layer normalization component (optional).
        """
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


class FeedForward(nn.Module):
    """Simple Feed Forward Network"""
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