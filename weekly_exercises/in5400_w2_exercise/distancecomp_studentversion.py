# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import torch
import time
import numpy as np


def forloopdists(feats, protos):
    dist = np.empty(shape=(feats.shape[0], protos.shape[0]), dtype=np.float)  # shape=(5000, 500)

    for i in range(feats.shape[0]):             # 5000 samples
        for j in range(protos.shape[0]):        # 500 samples

            # here @ is a np.dot function, because the elements are vectors
            dist[i][j] = (feats[i, :] - protos[j, :]).T @ (feats[i, :] - protos[j, :])  # (x - y).T @ (x - y)

    return dist


def numpydists(feats, protos):
    # here @ is a np.matmul function because the elements are matrices
    mid_factorization = -2 * feats @ protos.T                           # -2 * (X  @ Y.T)  shape=(5000,500)
    x_factorization = np.sum(feats**2, axis=1)[:, np.newaxis]           # (X @ X.T) = ||X||^2  shape=(5000, 1)
    y_factorization = np.sum(protos**2, axis=1)[np.newaxis, :]          # (Y @ Y.T) = ||Y||^2  shape=(1, 500)
    result = x_factorization + mid_factorization + y_factorization      # ||X||^2 -2 * (X  @ Y.T) + ||Y||^2
    return result


def pytorchdists(feats0, protos0, device):
    X, Y = torch.as_tensor(feats0, device=device), torch.as_tensor(protos0, device=device)
    mid_factorization = -2 * torch.mm(X, Y.T)
    x_factorization = torch.sum(torch.pow(X, 2), dim=1).unsqueeze(dim=1)
    y_factorization = torch.sum(torch.pow(Y, 2), dim=1).unsqueeze(dim=0)
    result = x_factorization + mid_factorization + y_factorization
    return result.data.numpy()


if __name__ == '__main__':
    feats = np.random.normal(size=(5000, 300))
    protos = np.random.normal(size=(500, 300))

    since = time.time()
    dists0 = forloopdists(feats, protos)
    time_elapsed = float(time.time()) - float(since)
    print('Comp complete in {:.3f}s'.format(time_elapsed))

    device = torch.device('cpu')
    since = time.time()
    dists1 = pytorchdists(feats, protos, device)
    time_elapsed = float(time.time()) - float(since)

    print('Comp complete in {:.3f}s'.format(time_elapsed))
    print(dists1.shape)

    # print('df0',np.max(np.abs(dists1-dists0)))
    since = time.time()
    dists2 = numpydists(feats, protos)
    time_elapsed = float(time.time()) - float(since)

    print('Comp complete in {:.3f}s'.format(time_elapsed))
    print(dists2.shape)
    print('df', np.max(np.abs(dists1 - dists2)))
