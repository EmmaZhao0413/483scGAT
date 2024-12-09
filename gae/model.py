'''GAE models'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from gae.layers import GraphConvolution


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj)

    def forward(self, x, adj, encode=False):
        z = self.encode(x, adj)
        return z, z, None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphAttentionLayer(nn.Module):
    """
    Custom implementation of a single Graph Attention Layer.
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.concat = concat

        self.W = Parameter(torch.empty(size=(in_features, out_features))) #nn.Linear(in_features, out_features, bias=False)
        self.a = Parameter(torch.empty(size=(2 * out_features, 1)))#nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        # h = F.dropout(h, self.dropout, self.training)
        # support = torch.mm(h, self.W)
        # output = torch.spmm(adj, support)
        # output = F.relu(output)
        # return output



        h = F.dropout(h, self.dropout, self.training)
        Wh = torch.mm(h, self.W)  # Linear transformation (dense)
        # print("h:"+str(h))
        # Attention mechanism (sparse handling)
        edge_indices = adj._indices()  # Get non-zero indices of sparse adj
        edge_values = adj._values()    # Get corresponding edge weights
        
        Wh_edge = torch.cat([Wh[edge_indices[0]], Wh[edge_indices[1]]], dim=1)  # Pair-wise concatenation
        e = self.leakyrelu(torch.matmul(Wh_edge, self.a).squeeze())  # Compute attention scores

        temperature = 0.5  # Adjust as needed (smaller values increase focus)
        attention_raw = torch.sparse_coo_tensor(edge_indices, e / temperature, adj.size())
        attention = torch.sparse.softmax(attention_raw, dim=1)


        # attention_norm = F.softmax(e, dim=0)  
        # attention_scaled = attention_norm * torch.sum(edge_values)
        # attention = torch.sparse_coo_tensor(edge_indices, attention_scaled, adj.size())


        h_prime = torch.sparse.mm(attention, Wh)  # Sparse matrix multiplication

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime
            
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GATModelVAE(nn.Module):
    """
    Variational Autoencoder model using custom GAT layers.
    """
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GATModelVAE, self).__init__()
        
        self.gat1 = GraphAttentionLayer(input_feat_dim, hidden_dim1, dropout, alpha=0.2, concat=True)
        self.gat2 = GraphAttentionLayer(hidden_dim1, hidden_dim2, dropout, alpha=0.2, concat=False)
        self.gat3 = GraphAttentionLayer(hidden_dim1, hidden_dim2, dropout, alpha=0.2, concat=False)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gat1(x, adj)
        return self.gat2(hidden1, adj), self.gat3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class GraphAttentionLayerMultiHead(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.6, concat=True):
        super(GraphAttentionLayerMultiHead, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat  # Whether to concatenate or average head outputs
        self.dropout = dropout

        # Multi-head layers
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, concat=True)
            for _ in range(num_heads)
        ])
    
    def forward(self, h, adj):
        # Apply each attention head
        head_outputs = [attn(h, adj) for attn in self.attention_heads]

        if self.concat:
            # Concatenate along feature dimension for intermediate layers
            return torch.cat(head_outputs, dim=1)
        else:
            # Average outputs for final layer
            return torch.mean(torch.stack(head_outputs, dim=0), dim=0)


class MultiHeadGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=5, dropout=0.6):
        super(MultiHeadGAT, self).__init__()
        self.gat1 = GraphAttentionLayerMultiHead(input_dim, hidden_dim, num_heads, dropout, concat=True)
        self.gat2 = GraphAttentionLayerMultiHead(hidden_dim * num_heads, output_dim, 1, dropout, concat=False)  # 1 head in final layer
        self.gat3 = GraphAttentionLayerMultiHead(hidden_dim * num_heads, output_dim, 1, dropout, concat=False)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gat1(x, adj)
        return self.gat2(hidden1, adj), self.gat3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar