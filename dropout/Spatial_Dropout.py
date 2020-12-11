# -*- coding:utf-8 -*-

import torch.nn as nn
from itertools import repeat

class Spatial_Dropout(nn.Module):
    """Spatial 1D version of Dropout.

     This version performs the same function as Dropout, however it drops
     entire 1D feature maps instead of individual elements. If adjacent frames
     within feature maps are strongly correlated (as is normally the case in
     early convolution layers) then regular dropout will not regularize the
     activations and will otherwise just result in an effective learning rate
     decrease. In this case, SpatialDropout1D will help promote independence
     between feature maps and should be used instead.

     Arguments:
       rate: Float between 0 and 1. Fraction of the input units to drop.

     Call arguments:
       inputs: A 3D tensor.
       training: Python boolean indicating whether the layer should behave in
         training mode (adding dropout) or in inference mode (doing nothing).

     Input shape:
       3D tensor with shape:
       `(samples, timesteps, channels)`

     Output shape:
       Same as input.

     References:
       - [Efficient Object Localization Using Convolutional
         Networks](https://arxiv.org/abs/1411.4280)
     """
    def __init__(self, drop_prob):
        super(Spatial_Dropout,self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self,input):
        return input.new().resize_(input.size(0),*repeat(1, input.dim() - 2),input.size(2))


