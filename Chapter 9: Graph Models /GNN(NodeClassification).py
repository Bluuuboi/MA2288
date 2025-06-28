#Reference: https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing#scrollTo=9r_VmGMukf5R

import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

torch.manual_seed(42)
dataset = Planetoid(root='./Planetoid', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]
print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


#Defining a Multi-Layer Perceptron (MLP)
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

model = MLP(hidden_channels=16)
print(model)

#Training the MLP model
# loss_function = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x)
#     loss = loss_function(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#     return loss
#
# def test():
#     model.eval()
#     out = model(data.x)
#     pred = torch.argmax(out, dim=1)
#     test_correct = pred[data.test_mask]==data.y[data.test_mask]
#     test_acc = int(test_correct.sum())/int(data.test_mask.sum())
#     return test_acc

# for epoch in range(201):
#     loss = train()
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
#
# test_acc = test()
# print(f'Test Accuracy: {test_acc:.4f}')
#Accuracy is only 0.5660 using MLP

torch.manual_seed(42)
#Now, we switch to using GNN
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GNN(hidden_channels=64)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_function = torch.nn.CrossEntropyLoss()

#Visualizing the nodes of our untrained GCN network, using TSNE, to embed our 7-dimensional node embeddings onto a 2D plane
def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)

def train():
    model.train()
    optimizer.zero_grad() #Clear gradients
    out = model(data.x, data.edge_index) #Perform a single forward pass
    loss = loss_function(out[data.train_mask], data.y[data.train_mask]) #Compute the loss solely based on the training nodes
    loss.backward() #Derive gradients
    optimizer.step() #Update parameters based on gradients
    return loss

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = torch.argmax(out, dim=1) #Use the class with the highes probability
    test_correct = pred[data.test_mask]==data.y[data.test_mask] #Check against ground-truth labels
    test_acc = int(test_correct.sum())/int(data.test_mask.sum()) #Derive ratio of correct predictions
    return test_acc

for epoch in range(201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')

#Visualizing the output embeddings of our trained model
model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)

