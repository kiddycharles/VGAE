import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VGAE(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim=None, num_class=None, alpha=None,
                 dropout=0.0):
        super(VGAE, self).__init__()
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)
        self.hidden2_dim = hidden2_dim
        self.mean = None
        self.logstd = None

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class VGAE2(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim=None, num_class=None, alpha=None,
                 dropout=0.0):
        super(VGAE2, self).__init__()
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)
        self.fc1 = nn.Linear(hidden2_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hidden2_dim = hidden2_dim
        self.mean = None
        self.logstd = None

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        Z = self.leaky(self.fc1(Z))
        A_pred = self.sigmoid(self.fc2(Z))
        return A_pred


class VGAE3(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim=None, num_class=None, alpha=None,
                 dropout=0.0):
        super(VGAE3, self).__init__()
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)
        self.fc1 = nn.Linear(hidden2_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.hidden2_dim = hidden2_dim
        self.mean = None
        self.logstd = None

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        Z = self.relu(self.fc1(Z))
        Z = self.relu(self.fc2(Z))
        A_pred = self.sigmoid(self.fc3(Z))
        return A_pred


class VGAE4(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim=None, num_class=None, alpha=None,
                 dropout=0.0):
        super(VGAE4, self).__init__()
        self.base_gcn = GraphAttentionLayer(input_dim, hidden1_dim, adj, dropout=dropout, alpha=alpha)
        self.gcn_mean = GraphAttentionLayer(hidden1_dim, hidden2_dim, adj, dropout=dropout, alpha=alpha)
        self.gcn_logstddev = GraphAttentionLayer(hidden1_dim, hidden2_dim, adj, dropout=dropout, alpha=alpha)
        self.fc1 = nn.Linear(hidden2_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, adj.shape[0])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hidden2_dim = hidden2_dim
        self.mean = None
        self.logstd = None

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        Z = self.relu(self.fc1(Z))
        Z = self.relu(self.fc2(Z))
        A_pred = self.sigmoid(self.fc3(Z))
        return A_pred


class VGAE5(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim=None, num_class=None, alpha=None,
                 dropout=0.0):
        super(VGAE5, self).__init__()
        self.base_gcn = GraphConvSparse(input_dim + num_class, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)
        self.fc1 = nn.Linear(hidden2_dim + num_class, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.hidden2_dim = hidden2_dim
        self.mean = None
        self.logstd = None

    def encode(self, X, Y):
        X = torch.cat([X.to_dense(), Y], dim=1)
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X, Y):
        Z = self.encode(X, Y)
        Z = torch.cat([Z, Y], dim=1)
        Z = self.relu(self.fc1(Z))
        A_pred = self.sigmoid(self.fc2(Z))
        return A_pred


class VGAE6(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim=None, num_class=None, alpha=None,
                 dropout=0.0):
        super(VGAE6, self).__init__()
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim + num_class, hidden2_dim, adj,
                                        activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(hidden1_dim + num_class, hidden2_dim, adj,
                                             activation=lambda x: x)
        self.fc1 = nn.Linear(hidden2_dim + num_class, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.hidden2_dim = hidden2_dim
        self.mean = None
        self.logstd = None

    def encode(self, X, Y):
        hidden = self.base_gcn(X)
        hidden = torch.cat([hidden, Y], dim=1)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X, Y):
        Z = self.encode(X, Y)
        Z = torch.cat([Z, Y], dim=1)
        Z = self.relu(self.fc1(Z))
        A_pred = self.sigmoid(self.fc2(Z))
        return A_pred


class VGAE7(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim=None, num_class=None, alpha=None,
                 dropout=0.0):
        super(VGAE7, self).__init__()
        self.base_gcn = GraphAttentionLayer(input_dim, hidden1_dim, adj, dropout=dropout, alpha=alpha)
        self.gcn_mean = GraphAttentionLayer(hidden1_dim + num_class, hidden2_dim, adj, dropout=dropout, alpha=alpha)
        self.gcn_logstddev = GraphAttentionLayer(hidden1_dim + num_class, hidden2_dim, adj, dropout=dropout,
                                                 alpha=alpha)
        self.fc1 = nn.Linear(hidden2_dim + num_class, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.hidden2_dim = hidden2_dim
        self.mean = None
        self.logstd = None

    def encode(self, X, Y):
        hidden = self.base_gcn(X)
        hidden = torch.cat([hidden, Y], dim=1)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X, Y):
        Z = self.encode(X, Y)
        Z = torch.cat([Z, Y], dim=1)
        Z = self.relu(self.fc1(Z))
        A_pred = self.sigmoid(self.fc2(Z))
        return A_pred


class VGAE8(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim=None, num_class=None, alpha=None,
                 dropout=0.0):
        super(VGAE8, self).__init__()
        self.base_gcn = GraphAttentionLayer(input_dim + num_class, hidden1_dim, adj, dropout=dropout, alpha=alpha)
        self.gcn_mean = GraphAttentionLayer(hidden1_dim, hidden2_dim, adj, dropout=dropout, alpha=alpha)
        self.gcn_logstddev = GraphAttentionLayer(hidden1_dim, hidden2_dim, adj, dropout=dropout, alpha=alpha)
        self.fc1 = nn.Linear(hidden2_dim + num_class, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.hidden2_dim = hidden2_dim
        self.mean = None
        self.logstd = None

    def encode(self, X, Y):
        X = torch.cat([X.to_dense(), Y], dim=1)
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X, Y):
        Z = self.encode(X, Y)
        Z = torch.cat([Z, Y], dim=1)
        Z = self.relu(self.fc1(Z))
        A_pred = self.sigmoid(self.fc2(Z))
        return A_pred


class VGAE9(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim=None, num_class=None, alpha=None,
                 dropout=0.0):
        super(VGAE9, self).__init__()
        self.base_gcn = GraphConvSparse(input_dim + num_class, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim + num_class, hidden2_dim, adj,
                                        activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(hidden1_dim + num_class, hidden2_dim, adj,
                                             activation=lambda x: x)
        self.fc1 = nn.Linear(hidden2_dim + num_class, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.hidden2_dim = hidden2_dim
        self.mean = None
        self.logstd = None

    def encode(self, X, Y):
        X = torch.cat([X.to_dense(), Y], dim=1)
        hidden = self.base_gcn(X)
        hidden = torch.cat([hidden, Y], dim=1)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X, Y):
        Z = self.encode(X, Y)
        Z = torch.cat([Z, Y], dim=1)
        Z = self.relu(self.fc1(Z))
        A_pred = self.sigmoid(self.fc2(Z))
        return A_pred


class VDGAE(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, num_class=None, alpha=None, dropout=0.0):
        super(VDGAE, self).__init__()
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.base_gcn2 = GraphConvSparse(hidden1_dim, hidden2_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden2_dim, hidden3_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(hidden2_dim, hidden3_dim, adj, activation=lambda x: x)
        self.fc1 = nn.Linear(hidden3_dim, hidden2_dim)
        self.fc2 = nn.Linear(hidden2_dim, hidden1_dim)
        self.fc3 = nn.Linear(hidden1_dim, adj.shape[0])
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hidden3_dim = hidden3_dim
        self.mean = None
        self.logstd = None

    def encode(self, X):
        hidden = self.base_gcn(X)
        hidden = self.base_gcn2(hidden)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden3_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        Z = self.leaky(self.fc1(Z))
        Z = self.leaky(self.fc2(Z))
        A_pred = self.sigmoid(self.fc3(Z))
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
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim=None, num_class=None, alpha=None,
                 dropout=0.0):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class VGAE4v2(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, hidden3_dim=None, num_class=None, alpha=None,
                 dropout=0.0):
        super(VGAE4v2, self).__init__()
        self.base_gcn = GraphAttentionV2Layer(input_dim, hidden1_dim, adj, n_heads=1, dropout=dropout)
        self.gcn_mean = GraphAttentionV2Layer(hidden1_dim, hidden2_dim, adj, n_heads=1, dropout=dropout)
        self.gcn_logstddev = GraphAttentionV2Layer(hidden1_dim, hidden2_dim, adj, n_heads=1, dropout=dropout)
        self.fc1 = nn.Linear(hidden2_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, adj.shape[0])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hidden2_dim = hidden2_dim
        self.mean = None
        self.logstd = None

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        Z = self.relu(self.fc1(Z))
        Z = self.relu(self.fc2(Z))
        A_pred = self.sigmoid(self.fc3(Z))
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


class GraphAttentionV2Layer(nn.Module):
    def __init__(self, in_features: int, out_features: int, adj, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.0,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        super(GraphAttentionV2Layer, self).__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        self.adj_mat = extend_adjacency_matrix(adj.to_dense(), n_heads=n_heads)
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        n_nodes = h.shape[0]
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)
        assert self.adj_mat.shape[0] == 1 or self.adj_mat.shape[0] == n_nodes
        assert self.adj_mat.shape[1] == 1 or self.adj_mat.shape[1] == n_nodes
        assert self.adj_mat.shape[2] == 1 or self.adj_mat.shape[2] == self.n_heads
        e = e.masked_fill(self.adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)


def extend_adjacency_matrix(adj_matrix, n_heads):
    n_nodes = adj_matrix.size(0)
    extended_adj_matrix = torch.zeros(n_nodes, n_nodes, n_heads)

    if n_heads == 1:
        extended_adj_matrix[:, :, 0] = adj_matrix
    else:
        for head in range(n_heads):
            extended_adj_matrix[:, :, head] = adj_matrix

    return extended_adj_matrix


def convert_scipy_csr_to_sparse_tensor(csr_matrix):
    coo_matrix = csr_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo_matrix.row, coo_matrix.col)).astype(np.int64))
    values = torch.from_numpy(coo_matrix.data.astype(np.float32))
    shape = torch.Size(coo_matrix.shape)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return sparse_tensor
