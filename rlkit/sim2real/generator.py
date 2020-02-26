"""
Generator model for ADDA.
Clone architecture of VAE in skew-fit
"""
import numpy as np
from torch import nn

from rlkit.torch.conv_networks import CNN
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.torch import pytorch_util as ptu
from rlkit.sim2real.discriminator import init_weights


class Encoder(nn.Module):
    """
    Generator model for target domain. Same with VAE's encoder but variance branch is removed, only
    keeping mean branch.
    """

    def __init__(self, representation_size,
                 architecture=imsize48_default_architecture,
                 imsize=48,
                 input_channels=3,
                 init_w=1e-3,
                 hidden_init=ptu.fanin_init):
        """Init generator."""
        super(Encoder, self).__init__()

        self.imsize = imsize
        self.input_channels = input_channels
        conv_args, conv_kwargs, deconv_args = \
            architecture['conv_args'], architecture['conv_kwargs'], architecture['deconv_args']
        conv_output_size = deconv_args['deconv_input_width'] * \
                           deconv_args['deconv_input_height'] * \
                           deconv_args['deconv_input_channels']

        self.encoder = CNN(
            **conv_args,
            paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
            input_height=self.imsize,
            input_width=self.imsize,
            input_channels=self.input_channels,
            output_size=conv_output_size,
            init_w=init_w,
            hidden_init=hidden_init,
            **conv_kwargs
        )

        self.fc1 = nn.Linear(self.encoder.output_size, representation_size)

        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)

        self.apply(init_weights)

    def forward(self, input):
        """Forward the generator (encoder of VAE)."""
        h = self.encoder(input)
        mu = self.fc1(h)
        return mu

