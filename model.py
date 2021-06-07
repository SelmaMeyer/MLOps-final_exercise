import torch.nn.functional as F
import torch

from torch import nn

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)  #(in_channels, out_channels, kernel)
        self.conv2 = nn.Conv2d(32, 64, 3)
    
        self.fc1 = nn.Linear(64*5*5, 128) # input should be: out_channels times image size 24*24 
        self.fc2 = nn.Linear(128, 10)

        # Add dropout
        self.dropout = nn.Dropout2d(0.25)

        # pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        # Pass data through conv layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Just print the shape to know which input should be given to the fc layer
        #print("SHAPE")
        #print(x.shape)

        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)

        # Pass data through fc1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Apply softmax to x
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
