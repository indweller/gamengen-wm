# custom_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MarioCNN(BaseFeaturesExtractor):
    """
    SB3-compatible feature extractor based on your original CNN.
    Output = 512-dimensional feature vector (same as your old 'linear').
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        # Same conv layers as your original PPO model
        self.conv1 = nn.Conv2d(n_input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # Determine flatten size using dummy pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            sample = sample / 255.0
            n_flatten = self._forward_conv(sample).shape[1]

        self.linear = nn.Linear(n_flatten, features_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, obs):
        obs = obs.float() / 255.0
        x = self._forward_conv(obs)
        x = self.linear(x)
        return x
