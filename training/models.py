#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class of the ML models.

@author: Matthijs de Jong
"""
# Standard library imports
from abc import ABC, abstractmethod

# Third-party library imports
import torch
from torch_geometric.nn import Linear



class Model(ABC, torch.nn.Module):
    """
    Base class for the models.
    """

    @abstractmethod
    def init_weights_normalized_normal(self, weight_init_std: float):
        """
        Initialize the weights of the network according to the normal
        distribution, but with the std divided by the number of in channels.
        The biases are initialized to zero.

        Parameters
        ----------
        weight_init_std : float
            The standard deviation of the normal distribution.
        """
        pass


class FCNN(Model):
    """
    Fully connected neural network. Consists of multiple feedforward layers.
    """

    def __init__(self,
                 LReLu_neg_slope: float,
                 weight_init_std: float,
                 size_in: int,
                 size_out: int,
                 N_layers: int,
                 N_node_hidden: int):
        """
        Parameters
        ----------
        LReLu_neg_slope : float
            The negative slope of the LReLu activation function.
        weight_init_std: float,
            The standard deviation of the normal distribution according to which the weights are initialized.
        N_layers : int
            The number of layers.
        N_node_hidden : int
            The number of nodes in the hidden layers.
        """
        super().__init__()

        # The activation function.
        self.LReLu_neg_slope = LReLu_neg_slope
        self.activation_f = torch.nn.LeakyReLU(LReLu_neg_slope)

        # The first layer
        self.lin_first = Linear(size_in, N_node_hidden)

        # Create the middle layers
        self.lin_middle_layers = torch.nn.ModuleList([Linear(N_node_hidden, N_node_hidden)
                                                      for _ in range(N_layers - 2)])

        # The last layer
        self.lin_last = Linear(N_node_hidden, size_out)

        # Initialize weights according to normal distribution
        self.init_weights_normalized_normal(weight_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the datapoint through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input of the model (i.e. the features).
        Returns
        -------
        x : torch.Tensor
            The output vector. Values should be in range (0,1).
        """
        # Passing the states through the consecutive linear (i.e. fully connected) layers
        x = self.lin_first(x)
        x = self.activation_f(x)

        for l in self.lin_middle_layers:
            x = l(x)
            x = self.activation_f(x)

        x = self.lin_last(x)
        x = torch.sigmoid(x)

        return x

    def init_weights_normalized_normal(self, weight_init_std: float):
        """
        Initialize the weights of this network according to the normal
        distribution, but with the std divided by the number of in channels.
        The biases are initialized to zero.

        Parameters
        ----------
        weight_init_std : float
            The standard deviation of the normal distribution.
        """

        def layer_weights_normal(m):
            """
            Apply initialization to a single layer.
            """
            classname = m.__class__.__name__
            # for every Linear layer in a model.
            if classname.find('Linear') != -1:
                std = weight_init_std / m.in_channels
                torch.nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(layer_weights_normal)
