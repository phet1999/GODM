import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MovingAverage(nn.Module):
    """移动平均层"""
    def __init__(self, kernel_size):
        super(MovingAverage, self).__init__()
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        return self.avg(x)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # 节点特征的线性变换层
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        :param x: 节点特征, 形状 [Batch, Num_Nodes, In_Features]
        :param adj: 批处理的邻接矩阵, 形状 [Batch, Num_Nodes, Num_Nodes]
        :return: 变换后的节点特征, 形状 [Batch, Num_Nodes, Out_Features]
        """
        # 批处理矩阵乘法: (B, N, N) @ (B, N, F_in) -> (B, N, F_in)
        support = torch.bmm(adj, x)
        
        # 应用线性变换: (B, N, F_in) -> (B, N, F_out)
        output = self.linear(support)
        
        return F.relu(output)

class GraphAttentionConstructor(nn.Module):
    
    """
    使用注意力机制动态构造图。
    这个模块从节点的输入特征X动态地为每个样本生成一个邻接矩阵。
    """
    def __init__(self, num_nodes, input_dim, attention_dim, k=None, use_softmax=True):
        """
        :param num_nodes: 节点数量
        :param input_dim: 输入节点特征的维度
        :param attention_dim: 用于计算注意力的Q和K的维度
        :param k: 可选参数，用于图稀疏化
        :param use_softmax: bool, 如果为True，使用softmax归一化注意力分数；
                                  如果为False，使用ReLU，这更接近原始实现，之后可以进行GCN归一化。
        """
        super(GraphAttentionConstructor, self).__init__()
        self.num_nodes = num_nodes
        self.k = k
        self.use_softmax = use_softmax
        self.attention_dim = attention_dim

        # 将输入特征映射到Query和Key空间的线性层
        self.W_q = nn.Linear(input_dim, attention_dim)
        self.W_k = nn.Linear(input_dim, attention_dim)
        
        # (可选) 仍然可以保留静态的节点嵌入，将其作为一种位置编码或身份信息
        # 这可以与输入特征拼接，提供更丰富的信息
        self.static_emb = nn.Parameter(torch.randn(num_nodes, attention_dim))

    def forward(self, x):
        """
        :param x: 节点特征矩阵, 维度 (batch_size, num_nodes, input_dim)
        :return: 归一化的邻接矩阵, 维度 (batch_size, num_nodes, num_nodes)
        """
        batch_size, _, _ = x.shape

        # 1. 投影到Query和Key空间
        # query/key shape: (batch_size, num_nodes, attention_dim)
        query = self.W_q(x)
        key = self.W_k(x)
        
        # 2. (可选) 加入静态嵌入，提供节点身份信息
        # 这使得模型不仅关注节点的动态特征，还关注节点本身是哪个
        query = query + self.static_emb.unsqueeze(0).expand(batch_size, -1, -1)
        key = key + self.static_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # 3. 计算注意力分数
        # attn_scores shape: (batch_size, num_nodes, num_nodes)
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.attention_dim)

        # 4. 根据use_softmax选择激活函数
        if self.use_softmax:
            # 使用Softmax得到一个行和为1的注意力矩阵，适合GAT类型的消息传递
            adj = F.softmax(attn_scores, dim=-1)
        else:
            # 使用ReLU，更像原始实现，可以捕获强连接，之后可以做GCN归一化
            adj = F.gelu(attn_scores)

        # 5. (可选) 稀疏化
        if self.k is not None and self.k < self.num_nodes:
            # 找到每行的 top-k 值
            topk_vals, _ = torch.topk(adj, self.k, dim=-1)
            # 创建一个阈值，小于该值的连接将被置零
            kth_value = topk_vals[:, :, -1].view(batch_size, self.num_nodes, 1)
            mask = (adj >= kth_value).float()
            adj = adj * mask
        
        # 6. GCN归一化 (注意：如果用了use_softmax=True，通常不再需要这一步)
        # 这一步主要在 use_softmax=False 的情况下使用
        if not self.use_softmax:
            adj = adj + torch.eye(self.num_nodes, device=adj.device).unsqueeze(0)
            d = adj.sum(dim=-1)
            d_inv_sqrt = torch.pow(d, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
            
            # A_norm = D^{-1/2} * A * D^{-1/2}
            normalized_adj = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
            return normalized_adj

        return adj  