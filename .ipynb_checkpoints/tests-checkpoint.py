import torch
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
from models.MultiHeadAttention import DeformableHeadAttention, generate_ref_points
import pdb

def test_deformable_attn():
        defomable_attn = DeformableHeadAttention(last_height = 16,last_width = 16, C=256, M=8, K=4, L = 1, dropout=0.1, return_attentions = True)
        defomable_attn = defomable_attn.cuda()
        w = 16
        h = 16
        querys = []
        ref_points = []
        for i in range(4):
            ww = w * 2**i
            hh = h * 2**i
            q = torch.rand([2, hh, ww, 256])
            q = q.cuda()
            querys.append(q)
            ref_point = torch.stack([ generate_ref_points(width=ww, height=hh) for _ in range(2)])
            ref_point = ref_point.type_as(q)
            ref_points.append(ref_point)
        pdb.set_trace()
        feat, attns = defomable_attn(querys[0], querys, ref_points[0])
        pdb.set_trace()
        #self.assertTrue(True)

if __name__ == '__main__':
    test_deformable_attn()