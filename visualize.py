#! /usr/bin/python3

"""Use torchview (https://github.com/mert-kurttutan/torchview) to visualize a network."""

import argparse
import math
import torch
import torch.nn as nn

from gameoflifegenerator import DataGenerator
from gameoflifemodels import (Net, BetterNet)

from torchview import draw_graph


def main():
    inparser = argparse.ArgumentParser(
        description="Arguments for the noise training script.")
    inparser.add_argument(
        '--steps', type=int, default=1,
        help='Number of steps to predict.')
    inparser.add_argument(
        '--m_factor', type=int, default=1,
        help='Overprovisioning factor for the neural network width.')
    inparser.add_argument(
        '--d_factor', type=float, default=1.,
        help='Overprovisioning factor for the neural network depth.')
    inparser.add_argument(
        '--presolve', default=False, action='store_true',
        help='Presolve the network weights.')
    inparser.add_argument(
        '--use_sigmoid', default=False, action='store_true',
        help='Use a sigmoid at the end of the network instead of the usual activation function.')
    inparser.add_argument(
        '--weight-init', default=False, action='store_true',
        help="Initialize weights in an Alexnet-like fashion. Alternating bias of 1 and 0, weights are normal with std=0.02.")
    inparser.add_argument(
        '--normalize', default=False, action='store_true',
        help="Normalize hidden layer outputs.")
    inparser.add_argument(
        '--activation_fun',
        required=False,
        default='ReLU',
        choices=['ReLU', 'ELU', 'LeakyReLU', 'RReLU', 'GELU'],
        type=str,
        help="Nonlinear activation function to use in the network.")
    inparser.add_argument(
        '--use_cuda', default=False, action='store_true',
        help='Use the default cuda device for model training.')
    inparser.add_argument(
        '--outfile', default="model", required=False,
        help="File to save the graph visualization.")
    args = inparser.parse_args()

    afun = getattr(torch.nn, args.activation_fun)
    if not args.weight_init and not args.normalize:
        net = Net(num_steps=math.floor(args.steps*args.d_factor), m_factor=args.m_factor, presolve=args.presolve, activation=afun,
                use_sigmoid=args.use_sigmoid)
        if args.use_cuda:
            net = net.cuda()
    else:
        net = BetterNet(num_steps=math.floor(args.steps*args.d_factor), m_factor=args.m_factor, activation=afun,
                weight_init=args.weight_init, normalize=args.normalize)
        if args.use_cuda:
            net = net.cuda()

    model_graph = draw_graph(net, input_size=(1, 5, 5), save_graph=True, filename=args.outfile,
            graph_dir='TB')



if __name__ == '__main__':
    main()

