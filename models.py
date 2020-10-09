import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)

        # self.gc2 = GraphConvolution(nhid1, nhid2)
        # self.gc3 = GraphConvolution(nhid2, nhid2)
        # self.gc4 = GraphConvolution(nhid2, nhid2)

        self.gc5 = GraphConvolution(nhid1, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc3(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc4(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc5(x, adj)
        return x
