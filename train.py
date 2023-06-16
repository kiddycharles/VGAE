
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import argparse
from input_data import load_data
from preprocessing import *
from model import *
import importlib
import os
import json


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "mps"


# Create the argument parser
parser = argparse.ArgumentParser(description='Script description')

# Add arguments for each variable
parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
parser.add_argument('--model', type=str, default='VGAE', help='Model name')
parser.add_argument('--input_dim', type=int, default=1433, help='Input dimension')
parser.add_argument('--hidden1_dim', type=int, default=32, help='Hidden layer 1 dimension')
parser.add_argument('--hidden2_dim', type=int, default=16, help='Hidden layer 2 dimension')
parser.add_argument('--hidden3_dim', type=int, default=8, help='Hidden layer 3 dimension')
parser.add_argument('--num_class', type=int, default=7, help='Number of classes')
parser.add_argument('--use_feature', type=bool, default=True, help='Use feature flag')
parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha')
parser.add_argument('--dropout', type=float, default=0.0, help='Drop out probability')


# Parse the arguments
args = parser.parse_args()

# Access the variables
dataset = args.dataset
existing_model = args.model
input_dim = args.input_dim
hidden1_dim = args.hidden1_dim
hidden2_dim = args.hidden2_dim
hidden3_dim = args.hidden3_dim
num_class = args.num_class
use_feature = args.use_feature
num_epoch = args.num_epoch
learning_rate = args.learning_rate

# Print the loaded variables
print(f"Loaded variables:\n"
      f"dataset: {dataset}\n"
      f"model: {existing_model}\n"
      f"input_dim: {input_dim}\n"
      f"hidden1_dim: {hidden1_dim}\n"
      f"hidden2_dim: {hidden2_dim}\n"
      f"hidden3_dim: {hidden3_dim}\n"
      f"num_class: {num_class}\n"
      f"use_feature: {use_feature}\n"
      f"num_epoch: {num_epoch}\n"
      f"learning_rate: {learning_rate}")

adj, features, labels = load_data(args.dataset)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create Model
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                    torch.FloatTensor(adj_norm[1]),
                                    torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                     torch.FloatTensor(adj_label[1]),
                                     torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                    torch.FloatTensor(features[1]),
                                    torch.Size(features[2]))

labels = torch.FloatTensor(labels)

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight


# Import the module dynamically based on the model name
module = importlib.import_module('model')
# init model and optimizer
existing_model = getattr(module, args.model)(adj_norm, args.input_dim, args.hidden1_dim, args.hidden2_dim,
                                             args.hidden3_dim, args.num_class, args.alpha, args.dropout)

optimizer = Adam(existing_model.parameters(), lr=args.learning_rate)


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


model_results = {'dataset': args.dataset, 'existing_model': args.model, 'input_dim':args.input_dim,
                 'hidden1_dim': args.hidden1_dim, 'hidden2_dim': args.hidden2_dim, 'hidden3_dim': args.hidden3_dim,
                 'num_class': args.num_class, 'use_feature': args.use_feature, 'num_epoch': args.num_epoch,
                 'learning_rate': args.learning_rate}

# train model
conditional_model = ['VGAE5', 'VGAE6', 'VGAE7', 'VGAE8', 'VGAE9']
val_rocs = []
val_aps = []
for epoch in range(args.num_epoch):
    t = time.time()
    if args.model in conditional_model:
        A_pred = existing_model(features, labels)
    else:
        A_pred = existing_model(features)
    # print(A_pred)
    optimizer.zero_grad()
    loss = log_lik = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
    if args.model != 'GAE':
        kl_divergence = 0.5 / A_pred.size(0) * (
                1 + 2 * existing_model.logstd - existing_model.mean ** 2 - torch.exp(existing_model.logstd) ** 2).sum(
            1).mean()
        loss -= kl_divergence

    loss.backward()
    optimizer.step()

    train_acc = get_acc(A_pred, adj_label)

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap),
          "time=", "{:.5f}".format(time.time() - t))
    val_aps.append(val_ap)
    val_rocs.append(val_roc)

test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))

model_results['test_roc'] = test_roc
model_results['test_ap'] = test_ap
model_results['val_rocs'] = val_rocs
model_results['val_aps'] = val_aps
# Directory path
directory = "results"


model_directory = os.path.join(directory, args.model)
# Assuming `model_results` is your dictionary with model results
# Create the directory if it doesn't exist
if not os.path.exists(model_directory):
    os.makedirs(model_directory, exist_ok=True)

# Save the results as a JSON file in the model subdirectory
result_path = os.path.join(model_directory, "result.json")
with open(result_path, "w") as file:
    json.dump(model_results, file)



# test_roc= 0.91393 test_ap= 0.92200  # VGAE   # VGAE original
# test_roc= 0.91186 test_ap= 0.91673  # VGAE2  # GCN + 2 layer MLP decoder
# test_roc= 0.90733 test_ap= 0.90612  # VGAE3  # GCN + 3 layer MLP decoder
# test_roc= 0.93413 test_ap= 0.92517  # VGAE4  # GAT only
# test_roc= 0.91566 test_ap= 0.91749  # VGAE5  # early concatenation + GCN
# test_roc= 0.92493 test_ap= 0.92690  # VGAE6  # late concatenation + GCN
# test_roc= 0.92642 test_ap= 0.92562  # VGAE7  # late concatenation + GAT
# test_roc= 0.91564 test_ap= 0.91889  # VGAE8  # early concatenation + GAT


# PubMed
# test_roc= 0.93873 test_ap= 0.94236 # VGAE
