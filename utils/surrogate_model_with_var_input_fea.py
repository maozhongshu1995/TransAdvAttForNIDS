import torch
import torch.nn as nn

class mlp_s_varfea(nn.Module):
    def __init__(self, fea_num):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(fea_num, 256), # layer 1
            nn.ReLU(),

            nn.Linear(256, 256), # layer 2
            nn.ReLU(),

            nn.Linear(256, 256), # layer 3
            nn.ReLU(),

            nn.Linear(256, 256), # layer 4
            nn.ReLU(),

            nn.Linear(256, 2) # classifier
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    net = mlp_s_varfea(80)
    x = torch.rand((128, 80))
    y = net(x)
    print(y.shape)