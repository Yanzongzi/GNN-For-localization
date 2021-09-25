from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
from scipy import io
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from data_processing import load_data
from models import GCN
import datetime

import matplotlib.pyplot as plt


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')

parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
#                     help='Weight decay (L2 loss on parameters).')

parser.add_argument('--hidden', type=int, default=2000,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

threshold = 1

seed = 42
anchor = 50
axis = 0
repeat = 20
loss_tem = np.zeros(repeat)

for axis in range(repeat):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    # Load data
    mode_fea, mode_adj, num_anchor, adj, features, labels, delta, degree, fea_original, fea_true, Range_Mat, Range, Dist_Mat, Dist, truncated_noise, idx_train, idx_val, idx_test = load_data(threshold, anchor)

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid1=args.hidden,
                nhid2=2000,
                nout=labels.shape[1],
                dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    print(model)

    loss_fun = torch.nn.MSELoss()

    if args.cuda:
        model.cuda()
        features = features.cuda()
        # features = Range_Mat.cuda()
        adj = adj.cuda()
        delta = delta.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = loss_fun(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        loss_val = loss_fun(output[idx_val], labels[idx_val])
        loss_val = torch.sqrt(loss_val)
        # loss_val = np.sqrt(loss_val)
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train (RMSE): {:.4f}'.format(loss_train.item()),
              # 'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val (RMSE): {:.4f}'.format(loss_val.item()),
              # 'acc_val: {:.4f}'.format(acc_val.item()),
              # 'time: {:.4f}s'.format(time.time() - t)
              )
        return loss_train


    def test():
        model.eval()
        output = model(features, adj)
        loss_test = loss_fun(output[idx_test], labels[idx_test])
        loss_test = torch.sqrt(loss_test)
        # loss_test = np.sqrt(loss_test)
        # acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f} (RMSE)".format(loss_test.item()),
              # "accuracy= {:.4f}".format(acc_test.item())
              )
        return output, loss_test


    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
    # for epoch in range(200):
        loss_train = train(epoch)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    predict, loss_test = test()
    loss_tem[axis] = loss_test.item()
    predict = predict.data.cpu().numpy()

    seed = seed + 1

loss = sum(loss_tem)/repeat
print("=====================================\n")
print("Averaged Test results:", "loss= {:.4f} (RMSE)".format(loss))

nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')  # Get the Now time

file_handle = open('result.txt', mode='a')
file_handle.write('=====================================\n')
file_handle.write(nowTime + '\n')
print(model, file=file_handle)
file_handle.write('feature mode (1 is filtered mode, 2 is sparse filtered mode):' + str(mode_fea) + '\n')
file_handle.write('adjacent mode (1 is filtered mode, 2 is sparse filtered mode):' + str(mode_adj) + '\n')
file_handle.write('loss_train (RMSE):' + format(loss_train.item()) + '\n')
file_handle.write('loss_test (RMSE):' + format(loss) + '\n')
file_handle.close()

# io.savemat('predict_' + str(num_anchor) + 'anchor_2layer_' + str(nowTime) + '.mat', {'predict': predict})

labels = labels.data.cpu().numpy()

plt.figure(1)
plt.scatter(predict[:, 0], predict[:, 1], color='b')
plt.show()
# plt.savefig("one.eps", format='eps', dpi=1000)   # save eps format figure. Close plt.show(), if the figure is blank.


