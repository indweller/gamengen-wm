import torch
import torch.nn as nn

class RewardModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(RewardModel, self).__init__()
        self.reg_head = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, latent_stack):
        vel = self.reg_head(latent_stack)
        done = self.cls_head(latent_stack)
        return vel, done