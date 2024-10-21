import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_units, GCN_dropout_rate, num_classes=3):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_units)
        self.conv2 = GCNConv(hidden_units, num_classes)
        self.dropout_rate = GCN_dropout_rate

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_weights)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weights)
        return x

'''
# Example of initializing and using the GCN model

# Initialize the model
num_node_features = 262  # Example number of node features
hidden_units = 32        # Number of hidden units in the first GCN layer
dropout_rate = 0.2       # Dropout rate for regularization
num_classes = 3          # Number of output classes

# Create an instance of the GCN model
model = GCN(num_node_features=num_node_features, 
            hidden_units=hidden_units, 
            GCN_dropout_rate=dropout_rate, 
            num_classes=num_classes)

# Example input: Graph data (you will need to provide actual graph data for real usage)
# Assuming data is a torch_geometric.data.Data object with x, edge_index, and edge_attr
# data = ... (you need to load your graph data here)
# output = model(data)
# print(output)
'''