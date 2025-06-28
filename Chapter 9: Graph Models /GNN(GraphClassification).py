#Reference: https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

torch.manual_seed(42)
dataset = TUDataset(root='./TUDataset', name='MUTAG')

print()
print(f'Dataset: {dataset}')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]
print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print()

#Dataset has 188 graphs, and the task is to classify them into 2 classes
torch.manual_seed(42)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
print()

#Using PyG, batch multiple graphs into a single giant graph
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

#Implementing GNN Model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        #1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        #2. Readout layer
        x = global_mean_pool(x, batch) #[batch_size, hidden_channels)

        #Apply final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

model = GCN(hidden_channels=64)
print(model)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.015)

def train():
    model.train()
    for data in train_loader: #Iterate in batches over the training dataset
        output = model(data.x, data.edge_index, data.batch) #Perform a single forward pass
        loss = loss_function(output, data.y) #Compute the loss
        loss.backward() #Derive gradients
        optimizer.step() #Update parameters based on gradients
        optimizer.zero_grad() #Clear gradients

def test(loader):
    model.eval()
    correct = 0
    for data in loader: #Iterate in batches over training/test dataset
        output = model(data.x, data.edge_index, data.batch)
        pred = output.argmax(dim=1) #Use the class with the highest probability
        correct += int((pred==data.y).sum()) #Check against ground-truth labels
    return correct/len(loader.dataset) #Derive ratio of correct predictions

for epoch in range(201):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

