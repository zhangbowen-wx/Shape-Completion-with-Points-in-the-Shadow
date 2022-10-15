#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import torch
from torch import nn, einsum
from pytorch3d.ops.knn import knn_points
from models.utils import MLP_Res
from pointnet2_ops.pointnet2_utils import grouping_operation


class SkipTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(SkipTransformer, self).__init__()
        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, pos, key, query):
        """
        Args:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = knn_points(pos_flipped, pos_flipped, K=self.n_knn).idx.int()
        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, dim, n, n_knn
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding  #

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.conv_end(agg)

        return y + identity


if __name__ == "__main__":
    import time
    ST = SkipTransformer(16).cuda()
    pos = torch.rand(32,3,1024).cuda()
    key = torch.rand(32,16,1024).cuda()
    query = torch.rand(32,16,1024).cuda()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    out = ST(pos, key, query)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(out.shape)