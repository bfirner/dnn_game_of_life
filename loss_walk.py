#! /usr/bin/python3

import argparse
import math
import torch
import torch.nn as nn

from gameoflifegenerator import DataGenerator
from gameoflifemodels import (Net, BetterNet)


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
    '--batches', default=5000, type=int,
    help='Total number of batches for loss calculation.')
inparser.add_argument(
    '--batch_size', default=32, type=int,
    help='Number of examples in a batch.')
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
    '--resume_from',
    type=str,
    required=False,
    help='Model weights to restore.')
inparser.add_argument(
    '--destination_weights',
    type=str,
    required=False,
    help='Weights to walk towards.')
args = inparser.parse_args()

afun = getattr(torch.nn, args.activation_fun)
lr_scheduler = None
use_cuda = True
if not args.weight_init and not args.normalize:
    net = Net(num_steps=math.floor(args.steps*args.d_factor), m_factor=args.m_factor, presolve=args.presolve, activation=afun,
            use_sigmoid=args.use_sigmoid)
    if use_cuda:
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters())
else:
    net = BetterNet(num_steps=math.floor(args.steps*args.d_factor), m_factor=args.m_factor, activation=afun,
            weight_init=args.weight_init, normalize=args.normalize)
    if use_cuda:
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters())
    # Adam and other optimizers adjust the learning rate automatically. But let's say that we think
    # we know better. This is all hand-crafted hyperparameter optimization at its finest.
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.2)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8], gamma=0.1)

# Load existing model weights if requested.
if args.resume_from is not None:
    checkpoint = torch.load(args.resume_from)
    net.load_state_dict(checkpoint["model_dict"])
    optimizer.load_state_dict(checkpoint["optim_dict"])

# Find a random basis vector to take the walk.
# Convert the model parameters to a single vector to make it simpler to modify them. The vector can
# be converted back to parameters with torch.nn.util.vector_to_parameters(vector, parameters)
param_vector = torch.nn.utils.parameters_to_vector(net.parameters())

# Create a basis vector with random values and then scale so that the magnitude of the vector is 1.
# Now each addition of this basis vector with take a unit step along a plane.
basis_vector = torch.randn(param_vector.size())
basis_vector = (basis_vector / basis_vector.abs().sum()).cuda()

# 0 to 1 in 0.1 increments, then 11 to 30 in steps of 1
step_sizes = [0.0] + [0.1] * 10 + [1] * 29

if args.destination_weights is not None:
    # Store the currente weights, verify that the new state fits into the model, remember them as
    # the target, then restore the original weights.
    original_weights = torch.nn.utils.parameters_to_vector(net.parameters())
    checkpoint = torch.load(args.destination_weights)
    net.load_state_dict(checkpoint["model_dict"])
    target_weights = torch.nn.utils.parameters_to_vector(net.parameters())
    torch.nn.utils.vector_to_parameters(original_weights, net.parameters())

    diff_vector = target_weights - original_weights
    magnitude = diff_vector.abs().sum()
    basis_vector = (diff_vector / magnitude).cuda()

    # Go from the initial state to the destination in 30 steps.
    step_sizes = [0.]
    for i in range(1, 31):
        step_sizes.append(magnitude/30.)


loss_fn = torch.nn.BCELoss()

datagen = DataGenerator()


net = net.eval()

losses = []
distance = 0.
with torch.no_grad():
    for step in step_sizes:
        distance += step
        # Reset the RNG state so that the datagen returns the same batches each time.
        torch.random.manual_seed(0)
        for batch_num in range(args.batches):
            batch, labels = datagen.getBatch(batch_size=args.batch_size, dimensions=(10, 10), steps=args.steps)
            if use_cuda:
                out = net.forward(batch.cuda())
                loss = loss_fn(out, labels.cuda())
            else:
                out = net.forward(batch)
                loss = loss_fn(out, labels)
            with torch.no_grad():
                losses.append(loss.mean())
        # Record the loss statistics and reset the loss list
        last_mean_loss = sum(losses)/len(losses)
        losses = []
        print(f"Step {distance} loss {last_mean_loss}")
        # Take a step along the basis vector
        param_vector.add_(basis_vector*step)
        torch.nn.utils.vector_to_parameters(param_vector, net.parameters())


exit()
