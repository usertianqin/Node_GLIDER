import scipy.io
import numpy as np
import scipy.sparse
import torch
import csv
import json
from os import path
import pickle as pkl
from torch_geometric.datasets import WebKB

def load_fb100(data_dir, filename):
    mat = scipy.io.loadmat(f'{data_dir}/facebook100/{filename}.mat')
    A = mat['A']
    metadata = mat['local_info']
    feature_vals_all = np.empty((0, 6))
    for f in ['Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Caltech36', 'Brown11', 'Yale4', 'Texas80',
              'Bingham82', 'Duke14', 'Princeton12', 'WashU32', 'Brandeis99', 'Carnegie49']:

        metadata = metadata.astype(np.int)
        feature_vals = np.hstack(
            (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        feature_vals_all = np.vstack(
            (feature_vals_all, feature_vals)
        )

    
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        # feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        feat_onehot = label_binarize(feat_col, classes=np.unique(feature_vals_all[:, col]))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    label = torch.tensor(label)
    label = torch.where(dataset.label > 0, 1, 0)
    return edge_index, label, node_feat

def load_twitch(data_dir, filename):
    assert filename in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = f"{data_dir}/twitch/{filename}"
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{filename}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2]=="True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{filename}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{filename}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id:idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]
    
    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)), 
                                 (np.array(src), np.array(targ))),
                                shape=(n,n))
    features = np.zeros((n,3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    # features = features[:, np.sum(features, axis=0) != 0] # remove zero cols. not need for cross graph task
    new_label = label[reorder_node_ids]
    label = new_label
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    label = torch.tensor(label)
    
    return edge_index, label, node_feat

def load_webKB(data_dir, filename):
    graph = WebKB(root=data_dir, name=filename)
    print(graph[0])
    exit(0)
    A = graph.edges
    label = graph.y
    features = graph.X
    print("A:", A)
    
    #return A, label, features

def load_Ciation(data_dir, filename):
    mat = scipy.io.loadmat(f'{data_dir}/facebook100/{filename}.mat')
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata

load_webKB('/home/tqin/meta-learning-graph/data', 'Cornell')