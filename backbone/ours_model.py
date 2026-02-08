import sys

import torch
from torch import nn
from torchvision.ops import MLP
from .layers.revin import RevIN
# from .layers.Embed import SinusoidalPosEmb
from .layers.MA_GCN import *
import math
from .layers.Transformer_EncDec import Encoder, EncoderLayer, Encoder_ori, LinearEncoder, LinearEncoder_Multihead
from .layers.SelfAttention_Family import FullAttention, AttentionLayer, EnhancedAttention
from .layers.Embed import DataEmbedding_inverted


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)
    def forward(self, x):
        x, gate = self.fc(x).chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        
        assert self.head_dim * n_heads == out_features, "out_features must be divisible by n_heads"

        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        
        self.out_proj = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, adj):
        """
        x: [B, N, D]
        adj: [N, N] (0-1 Matrix)
        """
        B, N, D = x.shape
        
        # 1. Linear Projections --> [B, N, heads, head_dim]
        q = self.W_q(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. Attention Score: (Q @ K.T) / sqrt(d) -> [B, heads, N, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if adj is not None:
            mask = (adj == 0).unsqueeze(0).unsqueeze(0) # [1, 1, N, N]
            attn = attn.masked_fill(mask, -1e9)

        # 4. Softmax & Value Aggregation
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # [B, heads, N, N] @ [B, heads, N, head_dim] -> [B, heads, N, head_dim]
        out = torch.matmul(attn, v)
        
        # 5. Concat Heads & Output Projection
        out = out.transpose(1, 2).contiguous().view(B, N, self.out_features)
        out = self.out_proj(out)
        
        return out

class AdvancedGraphResBlock(nn.Module):
    def __init__(self, dim, t_dim, nodes, n_heads=4):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(t_dim, dim * 2))
        
        self.temporal_process = nn.Sequential(
            nn.Linear(dim, dim),
            GLU(dim),
            nn.Linear(dim, dim)
        )
        
        self.spatial_gat = GraphAttention(dim, dim, n_heads=n_heads)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, t_emb, adj_matrix):
        # x: [B, N, dim]
        B, N, D = x.shape
        
        t_params = self.time_mlp(t_emb).unsqueeze(1) # [B, 1, dim*2]
        scale, shift = t_params.chunk(2, dim=-1)
        
        res = x * (1 + scale) + shift
        
        x = x + self.temporal_process(self.norm1(res))
    
        x_norm = self.norm2(x)
        s_res = self.spatial_gat(x_norm, adj_matrix)
        
        x = x + self.dropout(s_res)
        
        return x


class AdvancedResBlock(nn.Module):
    def __init__(self, dim, t_dim, nodes):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(t_dim, dim * 2))
        
        self.temporal_process = nn.Sequential(
            nn.Linear(dim, dim),
            GLU(dim),
            nn.Linear(dim, dim)
        )
        self.spatial_mix = nn.Sequential(
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, t_emb):
        # x: [B, N, dim]
        B, N, D = x.shape
        
        t_params = self.time_mlp(t_emb).unsqueeze(1) # [B, 1, dim*2]
        scale, shift = t_params.chunk(2, dim=-1)
        res = x * (1 + scale) + shift
        
        x = x + self.temporal_process(self.norm1(res))
        
        # [B, D, N] -> mix -> [B, D, N]
        s_res = x.transpose(1, 2)
        s_res = self.spatial_mix(s_res)
        x = x + self.norm2(s_res.transpose(1, 2))
        
        return x

class OLinearAdvancedDenoisingNetwork(nn.Module):
    def __init__(
        self,
        enc_in,       # NUM_NODES (170)
        seq_len,      # PRED_LEN (24/96)
        cond_len,     # SEQ_LENGTH (96)
        q_mat,        # 变换矩阵
        q_out_mat,    # 输出矩阵
        d_model=256,
        num_blocks=4,
        p_cond = 0.2,
        pred_z = True,
        embed_size = 8,
        **kwargs
    ):
        super().__init__()
        self.seq_length = seq_len
        self.nodes = enc_in
        self.p_cond = p_cond
        self.pred_z = pred_z
        self.embed_size = embed_size
        
        if isinstance(q_mat, np.ndarray): q_mat = torch.from_numpy(q_mat).float()
        if isinstance(q_out_mat, np.ndarray): q_out_mat = torch.from_numpy(q_out_mat).float()
        self.register_buffer('q_mat', q_mat)
        self.register_buffer('q_out_mat', q_out_mat)

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model)
        )

        self.cond_q_proj = nn.Linear(q_mat.shape[-1], d_model)
        
        self.x_proj = nn.Linear(seq_len*embed_size, d_model)

        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self.res_layers = nn.ModuleList([
            AdvancedResBlock(d_model, d_model, enc_in) for _ in range(num_blocks)
        ])

        self.final_head = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, q_out_mat.shape[-2]) # 映射到 Q_out 的输入空间
        )

    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def forward(self, x, t, condition=None, train=True):
        B, T_pred, N = x.shape

        if train and self.p_cond > 0:
            mask = torch.rand(B, 1, 1, device=x.device) < self.p_cond
            mask_to_drop_1d = mask.squeeze()
            condition[mask_to_drop_1d] = 0.
        
        t_emb = self.time_mlp(t)

        c = condition.transpose(1, 2) # [B, N, T_hist]
        if self.pred_z:
            if self.q_mat.dim() == 2:
                c_q = torch.einsum('bnl,lk->bnk', c, self.q_mat)
            else:
                c_q = torch.einsum('bnl,nlk->bnk', c, self.q_mat)
        else:
            c_q = c
        c_feat = self.cond_q_proj(c_q) # [B, N, d_model]

        x = self.tokenEmb(x, self.embeddings)
        
        x_feat = self.x_proj(x.flatten(-2)) # [B, N, d_model]

        h = self.fusion_proj(torch.cat([x_feat, c_feat], dim=-1))

        for block in self.res_layers:
            h = block(h, t_emb)

        q_out_coeffs = self.final_head(h)
        
            
        return q_out_coeffs.transpose(1, 2) # [B, T_pred, N]