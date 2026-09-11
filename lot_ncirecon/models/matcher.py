"""Neural approximation of local-OT patch similarity."""

import torch.nn as nn
class LightDeepOTMatcher(nn.Module):
    """
    Lightweight patch-pair similarity regressor.

    The original first script trained this network with BCE labels 0/1, so the
    sigmoid output easily saturated to values close to 1.0. In this modified
    version, the network directly regresses the local-OT similarity score in
    [0, 1]. Therefore, a threshold such as 0.7350 is on the same scale as the
    second local OT script.
    """
    def __init__(self, input_size=4096 * 2, hidden_size=256, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_size, 1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =========================
# File utilities
