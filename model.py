import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch.sparse as sparse

import args


class VGAE(nn.Module):
    def __init__(self, adj):
        super(VGAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class VGAE2(nn.Module):
    def __init__(self, adj):
        super(VGAE2, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.fc1 = nn.Linear(args.hidden2_dim, args.hidden1_dim)
        self.fc2 = nn.Linear(args.hidden1_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        Z = self.leaky(self.fc1(Z))
        A_pred = self.sigmoid(self.fc2(Z))
        return A_pred


class VGAE3(nn.Module):
    def __init__(self, adj):
        super(VGAE3, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.fc1 = nn.Linear(args.hidden2_dim, args.hidden1_dim)
        self.fc2 = nn.Linear(args.hidden1_dim, args.input_dim)
        self.fc3 = nn.Linear(args.input_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        Z = self.relu(self.fc1(Z))
        Z = self.relu(self.fc2(Z))
        A_pred = self.sigmoid(self.fc3(Z))
        return A_pred


class VGAE4(nn.Module):
    def __init__(self, adj):
        super(VGAE4, self).__init__()
        self.base_gcn = GraphAttentionLayer(args.input_dim, args.hidden1_dim, adj, dropout=0, alpha=0.2)
        self.gcn_mean = GraphAttentionLayer(args.hidden1_dim, args.hidden2_dim, adj, dropout=0, alpha=0.2)
        self.gcn_logstddev = GraphAttentionLayer(args.hidden1_dim, args.hidden2_dim, adj, dropout=0, alpha=0.2)
        self.fc1 = nn.Linear(args.hidden2_dim, args.hidden1_dim)
        self.fc2 = nn.Linear(args.hidden1_dim, args.input_dim)
        self.fc3 = nn.Linear(args.input_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        Z = self.relu(self.fc1(Z))
        Z = self.relu(self.fc2(Z))
        A_pred = self.sigmoid(self.fc3(Z))
        return A_pred


class VGAE5(nn.Module):
    def __init__(self, adj):
        super(VGAE5, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim + args.num_class, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.fc1 = nn.Linear(args.hidden2_dim + args.num_class, args.hidden1_dim)
        self.fc2 = nn.Linear(args.hidden1_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, X, Y):
        X = torch.cat([X.to_dense(), Y], dim=1)
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X, Y):
        Z = self.encode(X, Y)
        Z = torch.cat([Z, Y], dim=1)
        Z = self.relu(self.fc1(Z))
        A_pred = self.sigmoid(self.fc2(Z))
        return A_pred

class VGAE6(nn.Module):
    def __init__(self, adj):
        super(VGAE6, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim + args.num_class, args.hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim + args.num_class, args.hidden2_dim, adj, activation=lambda x: x)
        self.fc1 = nn.Linear(args.hidden2_dim + args.num_class, args.hidden1_dim)
        self.fc2 = nn.Linear(args.hidden1_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, X, Y):
        hidden = self.base_gcn(X)
        hidden = torch.cat([hidden, Y], dim=1)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X, Y):
        Z = self.encode(X, Y)
        Z = torch.cat([Z, Y], dim=1)
        Z = self.relu(self.fc1(Z))
        A_pred = self.sigmoid(self.fc2(Z))
        return A_pred


class VGAE7(nn.Module):
    def __init__(self, adj):
        super(VGAE7, self).__init__()
        self.base_gcn = GraphAttentionLayer(args.input_dim, args.hidden1_dim, adj, dropout=0, alpha=0.2)
        self.gcn_mean = GraphAttentionLayer(args.hidden1_dim + args.num_class, args.hidden2_dim, adj, dropout=0, alpha=0.2)
        self.gcn_logstddev = GraphAttentionLayer(args.hidden1_dim + args.num_class, args.hidden2_dim, adj, dropout=0, alpha=0.2)
        self.fc1 = nn.Linear(args.hidden2_dim + args.num_class, args.hidden1_dim)
        self.fc2 = nn.Linear(args.hidden1_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, X, Y):
        hidden = self.base_gcn(X)
        hidden = torch.cat([hidden, Y], dim=1)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X, Y):
        Z = self.encode(X, Y)
        Z = torch.cat([Z, Y], dim=1)
        Z = self.relu(self.fc1(Z))
        A_pred = self.sigmoid(self.fc2(Z))
        return A_pred



class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GAE(nn.Module):
    def __init__(self, adj):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
# 		return out


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, adj, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout
        self.adj = adj.to_dense()
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(self.adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
