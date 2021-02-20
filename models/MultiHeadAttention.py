
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_



def phi(width: int,height: int,p_q: torch.Tensor):
    new_point = p_q.clone().detach()
    new_point[..., 0] = new_point[..., 0] * (width - 1)
    new_point[..., 1] = new_point[..., 1] * (height - 1)

    return new_point

def generate_ref_points(width: int,
                        height: int):
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    grid_y = grid_y / (height - 1)
    grid_x = grid_x / (width - 1)

    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    return grid


class DeformableHeadAttention(nn.Module):
    """Deformable Attention Module"""
    def __init__(self,last_height,last_width, C, M=8, K=4, L = 1, dropout=0.1, return_attentions = False):
        """
        Args:
            - param C: emebedding size of the x's
            - param M: number of attention heads
            - param K: number of sampling points per attention head per feature level
            - param L: number of scale
            - param last_height: smallest feature height
            - param last_width: smallest feature width
            - param dropout: dropout ratio default =0.1,
            - param return_attentions: boolean, return attentions or not default = False
        """
        super(DeformableHeadAttention, self).__init__()
        assert C % M == 0 # check if C is divisible by M
        self.C_v = C // M
        self.M = M
        self.L = L
        self.K = K
        self.q_proj = nn.Linear(C, C)
        self.W_prim = nn.Linear(C, C)
        self.dimensions = [[ last_height * 2**i , last_width * 2**i] for i in range(self.L)]
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        # 2MLK for offsets MLK for A_mlqk
        self.delta_proj = nn.Linear(C, 2 * M * L * K) # delta p_q 2 *L* M * K
        self.Attention_projection = nn.Linear(C, M*K*L) # K probabilities per M and L

        self.W_m = nn.Linear(C, C)
        self.return_attentions = True
        self.init_parameters()
    def forward(self,z_q,Xs,p_q ,query_mask = None,x_masks = None):
        """
        Args:
        - param x_masks: batch, Height, Width
        - param query_mask: batch, H, W
        - param z_q: batch, H, W, C, query tensors
        - param Xs: List[batch, H, W, C] list of tensors representing multiscal image
        - param p_q: reference point 1 per pixel B, H, W, 2
        - return features                   Batch, Height, Width , C
                Attention                  Batch, Height, Width, L, M, K

        """
        #
        if x_masks is None:
            x_masks = [None] * len(Xs)

        output = {'attentions': None, 'deltas': None}

        B, H, W, _ = z_q.shape

        # B, H, W, C
        z_q = self.q_proj(z_q)

        # B, H, W, 2MLK
        deltas = self.delta_proj(z_q)
        # B, H, W, M, 2LK
        deltas = deltas.view(B, H, W, self.M, -1)

        # B, H, W, MLK
        A = self.Attention_projection(z_q)

        # put at - infinity probas masked (batch, H, W, 1)
        if query_mask is not None:
            query_mask_ = query_mask.unsqueeze(dim=-1)
            _, _, _, M_L_K = A.shape
            query_mask_ = query_mask_.expand(B, H, W, M_L_K)
            A = torch.masked_fill(A, mask=query_mask_, value=float('-inf'))

        # batch, H, W, M, L*K
        A = A.view(B, H, W, self.M, -1)
        A = F.softmax(A, dim=-1) # soft max over the L*K probabilities

        # mask nan position
        if query_mask is not None:
            # Batch, H, W, 1, 1
            query_mask_ = query_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            A = torch.masked_fill(A, query_mask_.expand_as(A), 0.0) # mask the possible nan values

        if self.return_attentions:
            output['attentions'] = A # # batch, H, W, M, L*K
            output['deltas'] = deltas # B, H, W, M, 2LK

        deltas = deltas.view(B, H, W, self.M, self.L, self.K, 2) # batch , H, W, M, L, K, 2
        deltas = deltas.permute(0, 3, 4, 5, 1, 2, 6).contiguous() # Batch, M, L, K, H, W, 2
        # Bacth * M, L, K, H, W, 2
        deltas = deltas.view(B * self.M, self.L, self.K, H, W, 2)

        A = A.permute(0, 3, 1, 2, 4).contiguous() # batch, M, H, W, L*K
        A = A.view(B * self.M, H * W, -1) # Batch *M, H*W, LK
        sampled_features_scale_list = []
        for l in range(self.L):
            x_l = Xs[l] # N H W C
            _, h, w, _ = x_l.shape

            x_l_mask = x_masks[l]

            # Batch, H, W, 2
            phi_p_q = phi(height=h, width=w, p_q=p_q) #phi multiscale
            # B, H, W, 2 -> B*M, H, W, 2
            phi_p_q = phi_p_q.repeat(self.M, 1, 1, 1) # repeat M points for every attention head
            # B, h, w, M, C_v
            W_prim_x = self.W_prim(x_l)
            W_prim_x = W_prim_x.view(B, h, w, self.M, self.C_v) # Separate the C features into M*C_v vectors
            #shape  batch, h( x_l ), w( x_l ), M, C_v

            if x_l_mask is not None: # si un masque est present
                # B, h, w, 1, 1
                x_l_mask = x_l_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
                x_l_mask = x_l_mask.expand(B, h, w, self.M, self.C_v)
                W_prim_x = torch.masked_fill(W_prim_x, mask=x_l_mask, value=0) # ne pas prendre en compte

            # Batch, M, C_v, h, w
            W_prim_x = W_prim_x.permute(0, 3, 4, 1, 2).contiguous()
            # Batch *M, C_v, h, w
            W_prim_x = W_prim_x.view(-1, self.C_v, h, w)
            # B*M, k, C_v, H, W
            sampled_features = self.compute_sampling(W_prim_x, phi_p_q, deltas, l, h, w)

            sampled_features_scale_list.append(sampled_features)

        # B*M, L, K, C_v, H, W
        #stack L (Batch *M, K, C_v, H, W) sampled features
        sampled_features_scaled = torch.stack(sampled_features_scale_list, dim=1)
        # B*M, H*W, C_v, LK
        sampled_features_scaled = sampled_features_scaled.permute(0, 4, 5, 3, 1, 2).contiguous()
        sampled_features_scaled = sampled_features_scaled.view(B * self.M, H * W, self.C_v, -1)
        # sampled_features_scaled (n B*M ,l H*W ,d C_v ,LK)
        # A (n B*M, l H*W ,s L*K)
        #result of the sum of product  (n B*M , l H*W, d C_v)  B*M, H*W, C_v
        Attention_W_prim_x_plus_delta = torch.einsum('nlds, nls -> nld', sampled_features_scaled, A)

        # B, M, H, W, C_v
        Attention_W_prim_x_plus_delta = Attention_W_prim_x_plus_delta.view(B, self.M, H, W, self.C_v)
        # B, H, W, M, C_v
        Attention_W_prim_x_plus_delta = Attention_W_prim_x_plus_delta.permute(0, 2, 3, 1, 4).contiguous()
        # B, H, W, M * C_v
        Attention_W_prim_x_plus_delta = Attention_W_prim_x_plus_delta.view(B, H, W, self.C_v * self.M)

        final_features = self.W_m(Attention_W_prim_x_plus_delta)
        if self.dropout:
            final_features = self.dropout(final_features)

        return final_features, output

    def compute_sampling(self, W_prim_x,phi_p_q, deltas, layer, h, w):
        offseted_features = []
        for k in range(self.K): # for K points
            phi_p_q_plus_deltas = phi_p_q + deltas[:, layer, k, :, :, :] # p_q + delta p_mqk
            vgrid_x = 2.0 * phi_p_q_plus_deltas[:, :, :, 0] / max(w - 1, 1) - 1.0 # copied
            vgrid_y = 2.0 * phi_p_q_plus_deltas[:, :, :, 1] / max(h - 1, 1) - 1.0 # copied
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3) # stack the

            # B*M, C_v, H, W
            # bilinear interpolation as explained in deformable convolution

            sampled = F.grid_sample(W_prim_x, vgrid_scaled, mode='bilinear', padding_mode='zeros')
            offseted_features.append(sampled)
        return torch.stack(offseted_features, dim=3)

    def init_parameters(self):
        torch.nn.init.constant_(self.delta_proj.weight, 0.0)
        torch.nn.init.constant_(self.Attention_projection.weight, 0.0)

        torch.nn.init.constant_(self.Attention_projection.bias, 1 / (self.L * self.K))

        def init_xy(bias, x, y):
            torch.nn.init.constant_(bias[:, 0], float(x))
            torch.nn.init.constant_(bias[:, 1], float(y))

        # caution: offset layout will be  M, L, K, 2
        bias = self.delta_proj.bias.view(self.M, self.L, self.K, 2)

        init_xy(bias[0], x=-self.K, y=-self.K)
        init_xy(bias[1], x=-self.K, y=0)
        init_xy(bias[2], x=-self.K, y=self.K)
        init_xy(bias[3], x=0, y=-self.K)
        init_xy(bias[4], x=0, y=self.K)
        init_xy(bias[5], x=self.K, y=-self.K)
        init_xy(bias[6], x=self.K, y=0)
        init_xy(bias[7], x=self.K, y=self.K)
