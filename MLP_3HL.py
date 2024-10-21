import torch
import torch.nn as nn

class MLP_3HL(nn.Module):
    def __init__(self, input_features, hidden_units_1, hidden_units_2, hidden_units_3, MLP_dropout_rate, num_classes=3):
        """
        Initializes a Multi-Layer Perceptron (MLP) with 3 hidden layers.

        Args:
            input_features (int): Number of input features.
            hidden_units_1 (int): Number of units in the first hidden layer.
            hidden_units_2 (int): Number of units in the second hidden layer.
            hidden_units_3 (int): Number of units in the third hidden layer.
            num_classes (int): Number of output classes for classification. Defaults to 3.
            MLP_dropout_rate (float): Dropout rate applied after each hidden layer. Defaults to 0.2.
        """
        super(MLP_3HL, self).__init__()

        # First hidden layer
        self.fc1 = nn.Linear(input_features, hidden_units_1, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_units_1)
        self.relu1 = nn.ReLU(inplace=False)
        self.dropout1 = nn.Dropout(p=MLP_dropout_rate)

        # Second hidden layer
        self.fc2 = nn.Linear(hidden_units_1, hidden_units_2, bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_units_2)
        self.relu2 = nn.ReLU(inplace=False)
        self.dropout2 = nn.Dropout(p=MLP_dropout_rate)

        # Third hidden layer
        self.fc3 = nn.Linear(hidden_units_2, hidden_units_3, bias=True)
        self.bn3 = nn.BatchNorm1d(hidden_units_3)
        self.relu3 = nn.ReLU(inplace=False)
        self.dropout3 = nn.Dropout(p=MLP_dropout_rate)

        # Output layer
        self.fc4 = nn.Linear(hidden_units_3, num_classes, bias=True)

    def forward(self, x):
        """
        Defines the forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output logits for each input.
        """
        # First hidden layer with conditional batch normalization
        x = self.fc1(x)
        if x.size(0) != 1:  # Only apply batch normalization when batch size > 1
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Second hidden layer with conditional batch normalization
        x = self.fc2(x)
        if x.size(0) != 1:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Third hidden layer with conditional batch normalization
        x = self.fc3(x)
        if x.size(0) != 1:
            x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Output layer
        x = self.fc4(x)
        return x

'''
# Example of initializing and using the MLP_3HL model

# Initialize the model
input_features = 262  # Example input feature size
hidden_units_1 = 256  # Number of units in the first hidden layer
hidden_units_2 = 128  # Number of units in the second hidden layer
hidden_units_3 = 64   # Number of units in the third hidden layer
dropout_rate = 0.2    # Dropout rate
num_classes = 3       # Number of output classes

# Create an instance of the MLP_3HL model
model = MLP_3HL(input_features=input_features, 
                hidden_units_1=hidden_units_1, 
                hidden_units_2=hidden_units_2, 
                hidden_units_3=hidden_units_3, 
                num_classes=num_classes, 
                MLP_dropout_rate=dropout_rate)

# Example input: Tensor of size (batch_size=32, input_features)
inputs = torch.randn(32, input_features)

# Pass the input data through the model to get predictions
outputs = model(inputs)

# Print the model's output
print(outputs)
'''