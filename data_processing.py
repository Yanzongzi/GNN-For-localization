from scipy.io import loadmat
import numpy as np
import scipy.sparse as sp
import torch


def normalize_fea(mx):
    """Row-normalize feature matrix"""
    fea_max = np.array(mx.sum(1))
    r_inv = np.power(fea_max, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def adjacent_normalize(mx):
    """"D^(-0.5)*A*D^(-0.5)"""
    rowsum = np.array(mx.sum(1))
    r_inv_half = np.power(rowsum, -0.5).flatten()
    r_inv_half[np.isinf(r_inv_half)] = 0.
    r_mat_inv = sp.diags(r_inv_half)
    mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(threshold, num_anchor):
    m = loadmat("./Networks/8anchor_1000agent_10PercentNLOS_mediumLOS.mat")

    Range_Mat = m["Range_Mat"]  # Range = Distance + noise
    Dist_Mat = m["Dist_Mat"]
    labels = m["nodes"]

    # Range_Mat = abs(Range_Mat)  # Get the absolute value because some negative existence
    length = Range_Mat.shape[0]
    Range = Range_Mat.copy()
    ## use threshold to truncate the distance matrix
    Range[Range > threshold] = 0

    # Get the adjacent matrix
    # mode1 of adjacent matrix (filtered connection matrix)
    mode_adj = 1
    Range_tem = Range.copy()
    Range_tem[Range_tem > 0] = 1
    Adj = Range_tem
    # print(Delta)

    # Get the feature matrix
    ## feature mode1 (filtered feature matrix)
    mode_fea = 1
    features = Range

    # Get the degree and Laplacian matrix
    Degree = np.sum(Adj, axis=1)
    Delta = np.diag(Degree) - Adj

    # get the truncated true feature
    Dist_tem = Dist_Mat.copy()
    Dist = np.multiply(Dist_tem, Adj)

    # truncated noise is the unnormalized truncated noize, which is used to compare with the full matrix noise
    truncated_noise = features - Dist

    # Sparse matrix form
    features = sp.csr_matrix(features, dtype=np.float64)
    Adj = sp.csr_matrix(Adj, dtype=np.float64)

    # Normalize
    # change Dist_Mat/Dist/Range_Mat to change the output of features_original in frequency_analysis.py
    features_original = normalize(Dist_Mat)
    features_true = normalize(Dist)
    # features_full = normalize(Range_Mat)

    #
    # features = normalize_fea(features)
    # Range_Mat = normalize(Range_Mat)

    features = normalize(features)
    # adj = adjacent_normalize(Adj + sp.eye(Adj.shape[0]))
    # adj = normalize(adj)
    adj = normalize(Adj + sp.eye(Adj.shape[0]))

    idx_train = range(8, num_anchor+8)
    idx_val = range(num_anchor+8, 508)
    idx_test = range(num_anchor+8, 508)

    # features = torch.FloatTensor(features.todense())
    features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.FloatTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # delta = sparse_mx_to_torch_sparse_tensor(Delta)
    delta = torch.FloatTensor(Delta)
    degree = torch.FloatTensor(np.diag(Degree))
    fea_original = torch.FloatTensor(features_original)
    fea_true = torch.FloatTensor(features_true)
    truncated_noise = torch.FloatTensor(truncated_noise)
    Range_Mat = torch.FloatTensor(Range_Mat)
    Dist_Mat = torch.FloatTensor(Dist_Mat)
    Dist = torch.FloatTensor(Dist)
    Range = torch.FloatTensor(Range)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print(Delta)

    return mode_fea, mode_adj, num_anchor, adj, features, labels, delta, degree, fea_original, fea_true, Range_Mat, Range, Dist_Mat, Dist, truncated_noise, idx_train, idx_val, idx_test
