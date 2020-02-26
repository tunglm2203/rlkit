"""
Discriminator model for ADDA.
clone from: https://github.com/corenel/pytorch-adda/blob/master/models/discriminator.py
"""

from torch import nn


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

        # Ref: https://github.com/corenel/pytorch-adda/blob/96f2689dd418ef275fcd0b057e5dff89be5762c5/utils.py#L34
        self.apply(init_weights)

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)
