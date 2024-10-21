import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConditionalBatchNorm3d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features)

    def forward(self, x):
        if x.size(0) > 1:  
            return self.bn(x)
        return x
        
class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        if x.size(0) > 1:  
            return self.bn(x)
        return x
        
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = ConditionalBatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = ConditionalBatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], dim=1) 

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = ConditionalBatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        return self.pool(x)

class Custom_3D_DenseNet(nn.Module):
    def __init__(self, growth_rate, block_config, bn_size=, drop_rate,
                 hidden_units_1, hidden_units_2, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, 2*growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', ConditionalBatchNorm3d(2*growth_rate)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=2, stride=2)),
        ]))

        num_features = 2*growth_rate
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
            self.features.add_module('transition%d' % (i + 1), trans)
            num_features = num_features // 2
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features*2*2*2, hidden_units_1, bias=True),
            ConditionalBatchNorm1d(hidden_units_1),
            nn.ReLU(inplace=False),
            nn.Dropout(p=drop_rate),

            nn.Linear(hidden_units_1, hidden_units_2, bias=True),
            ConditionalBatchNorm1d(hidden_units_2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=drop_rate),
            
            nn.Linear(hidden_units_2, num_classes, bias=True)
        )

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out

'''
# Example of initializing and using the Custom_3D_DenseNet model

# Initialize the model
growth_rate = 32
block_config = (3, 6, 12, 8)  # Number of layers in each dense block
bn_size = 4                   # Batch norm size
drop_rate = 0.1               # Dropout rate
hidden_units_1 = 256          # Number of units in the first fully connected layer
hidden_units_2 = 64           # Number of units in the second fully connected layer
num_classes = 3               # Number of output classes

# Create an instance of the Custom_3D_DenseNet model
model = Custom_3D_DenseNet(growth_rate=growth_rate, 
                           block_config=block_config, 
                           bn_size=bn_size, 
                           drop_rate=drop_rate, 
                           hidden_units_1=hidden_units_1, 
                           hidden_units_2=hidden_units_2, 
                           num_classes=num_classes)

# Print a summary of the model architecture using an input size of (1, 64, 64, 64)
summary(test_CustomDenseNet_model, input_size=(1, 64, 64, 64))

# Example input: 3D tensor (batch_size=32, channels=1, depth=32, height=32, width=32)
inputs = torch.randn(32, 1, 64, 64, 64)

# Pass the input data through the model to get predictions
outputs = model(inputs)

# Print the model's output
print(outputs)
'''