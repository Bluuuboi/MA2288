#Reference: https://machinelearningmastery.com/a-gentle-introduction-to-graph-neural-networks-in-python/

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx

torch.manual_seed(42)
#Define Graph dataset
#Edge index is a matrix of edges/connections between 5 users (User 0-4)
#First connection is from User 0 to User 1, Second connection is the reciprocal of the previous one: User 1 to 0
# So on for the next connections
edge_index = torch.tensor([
    [0, 1, 0, 2, 0, 4, 2, 4],
    [1, 0, 2, 0, 4, 0, 4, 2],
], dtype=torch.long)

#Now, we model two numerical features for each user; age and their interest in sports (1 as having interest, 0 as no interest)
#Define data features
node_features = torch.tensor([
    [25, 1],
    [30, 0],
    [22, 1],
    [35, 0],
    [27, 1],
], dtype=torch.float)

#Visualize our graph
#Convert the edge_index tensor to a list of edge tuples
edge_list = edge_index.t().tolist()

#Create a Networkx graph from the edge list
G = nx.Graph()
G.add_edges_from(edge_list)

#Optionally, include nodes that might be isolated (i.e. User 3, since he/she has no connections)
G.add_nodes_from(range(node_features.size(0)))

#Generate a layout for the nodes
pos = nx.spring_layout(G, seed=42)

#Draw the graph
plt.figure(figsize=(10, 10))
nx.draw_networkx(G, pos=pos, with_labels=True, node_size=800, node_color='lightblue')
plt.title('Visualization of the Social Network Graph')
plt.axis('off')
# plt.show()

#Now we build a GNN
#Define dataset labels (whether user has >= 2 friends)
num_friends = torch.tensor([3, 1, 2, 0, 3])
labels = (num_friends >= 2).long()

# Mask for separating training and testing data
train_mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)
data = Data(x=node_features, edge_index=edge_index, y=labels, train_mask=train_mask)
# print(data)
# print(train_mask)
# print(data.train_mask)

#Define model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GNN(input_dim=2, hidden_dim=12, output_dim=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    predictions = model(data).argmax(dim=1)

print("\nFinal Predictions (1=Popular, 0=Not Popular):", predictions.tolist())


