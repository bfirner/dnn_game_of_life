#! /usr/bin/python3

import argparse
import math
import torch
import torch.nn as nn

from gameoflifegenerator import DataGenerator
from gameoflifemodels import (Net, BetterNet)


def main():
    inparser = argparse.ArgumentParser(
        description="Arguments for the noise training script.")
    inparser.add_argument(
        '--steps', type=int, default=1,
        help='Number of steps to predict.')
    inparser.add_argument(
        '--seed', type=int, default=1,
        help='Seed for the random number generator.')
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
        help='Total number of training batches.')
    inparser.add_argument(
        '--batch_size', default=32, type=int,
        help='Number of examples in a batch.')
    inparser.add_argument(
        '--demo', default=False, action='store_true',
        help='Just print out some examples from the game of life.')
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
        '--use_cuda', default=False, action='store_true',
        help='Use the default cuda device for model training.')
    args = inparser.parse_args()

    torch.random.manual_seed(args.seed)

    afun = getattr(torch.nn, args.activation_fun)
    lr_scheduler = None
    if not args.weight_init and not args.normalize:
        net = Net(num_steps=math.floor(args.steps*args.d_factor), m_factor=args.m_factor, presolve=args.presolve, activation=afun,
                use_sigmoid=args.use_sigmoid)
        if args.use_cuda:
            net = net.cuda()
        optimizer = torch.optim.Adam(net.parameters())
    else:
        net = BetterNet(num_steps=math.floor(args.steps*args.d_factor), m_factor=args.m_factor, activation=afun,
                weight_init=args.weight_init, normalize=args.normalize)
        if args.use_cuda:
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

    loss_fn = torch.nn.BCELoss()

    datagen = DataGenerator()

    if args.demo:
        batch, labels = datagen.getBatch(batch_size=args.batch_size, dimensions=(3, 3), steps=args.steps)

        for idx in range(args.batch_size):
            print(f"{batch[idx]}")
            print(f"=>")
            print(f"{labels[idx]}")
            print()
            print()
            print()
        exit()


    losses = []
    for batch_num in range(args.batches):
        batch, labels = datagen.getBatch(batch_size=args.batch_size, dimensions=(10, 10), steps=args.steps)
        optimizer.zero_grad()
        if args.use_cuda:
            out = net.forward(batch.cuda())
            loss = loss_fn(out, labels.cuda())
        else:
            out = net.forward(batch)
            loss = loss_fn(out, labels)
        with torch.no_grad():
            losses.append(loss.mean())
        if 0 == batch_num % 1000:
            # Record the loss statistics and reset the loss list
            last_mean_loss = sum(losses)/len(losses)
            losses = []
            with torch.no_grad():
                print(f"After {batch_num} mean loss was {last_mean_loss}")
                # This is a bit too spammy with larger networks.
                #if 1 == args.steps and args.m_factor == 1:
                if args.m_factor == 1:
                    for i, block in enumerate(net.net):
                        if type(block) is torch.nn.Sequential:
                            for j, layer in enumerate(block):
                                if hasattr(layer, 'bias') and layer.bias is not None and hasattr(layer, 'weight'):
                                    print(f"At batch {batch_num} layer {i},{j} has weights {layer.weight.tolist()} and bias {layer.bias.tolist()}")
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()


    print("Final layer weights are:")
    for i, layer in enumerate(net.net):
        if hasattr(layer, 'bias') and layer.bias is not None and hasattr(layer, 'weight'):
            print(f"Final layer {i} weights are {layer.weight.tolist()} and bias is {layer.bias.tolist()}")

    last_mean_loss = sum(losses)/len(losses)
    print(f"After {args.batches} mean loss was {last_mean_loss}")


    torch.save({
        "model_dict": net.state_dict(),
        "optim_dict": optimizer.state_dict(),
        }, "gol_model.pyt")

    # Show some results with the final network.
    batch, labels = datagen.getBatch(batch_size=10, dimensions=(5, 5), steps=args.steps)
    net = net.eval()
    if args.use_cuda:
        outputs = net(batch.cuda())
    else:
        outputs = net(batch)

    print("Final network results are:")
    for idx in range(10):
        print(f"Batch")
        print(f"{batch[idx]}")
        print(f"=>")
        print(f"NN Outputs")
        print(f"{outputs[idx]}")
        print(f"=>")
        print(f"Labels")
        print(f"{labels[idx]}")
        print()
        print()


    batch, labels = datagen.getBatch(batch_size=1000, dimensions=(5, 5), steps=args.steps)
    net = net.eval()
    if args.use_cuda:
        outputs = net(batch.cuda()).round()
        labels = labels.cuda()
    else:
        outputs = net(batch).round()
    matches = 0
    for idx in range(1000):
        # TODO Add an option that allows for close matches, not just exact
        if outputs[idx].equal(labels[idx]):
            matches += 1
    print(f"Success rate: {matches/1000.}")
    if 1000 == matches:
        print("Training success.")
    else:
        print("Training failure.")

if __name__ == '__main__':
    main()
