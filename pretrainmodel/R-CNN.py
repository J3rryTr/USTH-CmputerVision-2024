import torch.nn as nn

class RCNNs(nn.Module):
    def __init__(self):
        super(RCNNs, self).__init__()
        self.features = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(5,5), stride=4), #conv0
        nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(5,5), stride=4), #conv1
        nn.ReLU(inplace=True),
        nn.MaxPool2d(5, 4), #MaxPooling0

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=4), #conv2
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 4), #MaxPooling1

        nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=4), #conv3
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), stride=4), #conv4

        nn.Conv2d(in_channels=384, out_channels=256,kernel_size=(1,1), stride=4), #conv5
        nn.MaxPool2d(3, 4),

        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 1000),
        nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.features(x)
        return x

# checking architecture
model = RCNNs()
print(model)