#! /usr/bin/python3

import argparse
import math
import torch
import torch.nn as nn

class Net(torch.nn.Module):
    """Network that demonstrates some fundamentals of fitting."""

    def __init__(self, num_steps=1, m_factor=1, presolve=True, activation=nn.ReLU, use_sigmoid=False):
        """Initialize the demonstration network with a single output.

        Arguments:
            num_steps              (int): There will be one additional 3x3 convolution per step.
            m_factor (int): Multiplication factor. Multiply number of feature maps by this amount.
            presolve (bool): Initialize weights and bias values to the solution if True.
            activation (constructor): The class constructor for the activation function to use.
        """
        super(Net, self).__init__()
        self.net = torch.nn.ModuleList()

        with torch.no_grad():
            self.weighted_sum = torch.tensor([[0.,1.,0.], [1.,0.5,1.], [0.,1.,0.]])

            for step in range(num_steps):
                last_step = (step+1 == num_steps)

                block = []
                block.append(
                    nn.Conv2d(
                        in_channels=1, out_channels=2*m_factor, kernel_size=3, stride=1, padding=1))
                if presolve:
                    block[-1].weight[0][0] = self.weighted_sum
                    block[-1].weight[1][0] = self.weighted_sum
                    block[-1].bias[0] = -2
                    block[-1].bias[1] = -3.5
                block.append(activation())

                block.append(
                    nn.Conv2d(
                        in_channels=2*m_factor, out_channels=m_factor, kernel_size=1, stride=1, padding=0))
                # The final output should end up 0 when the second layer is 4 times the first. Use a
                # ratio of 1:5 instead of 1:4 to move the output away from 0.  The next layer will
                # set an exact output 1 using bias, so if the output should be alive make it
                # negative here. A negative weight at the next layer will map a positive output from
                # this layer to 0 once it passes through a ReLU.
                if presolve:
                    block[-1].weight[0][0] = -2
                    block[-1].weight[0][1] = 10
                    block[-1].bias[0] = 1
                block.append(activation())

                block.append(
                    nn.Conv2d(
                        in_channels=m_factor, out_channels=1, kernel_size=1, stride=1, padding=0))
                if presolve:
                    if use_sigmoid and last_step:
                        # Set things up for the sigmoid
                        block[-1].weight[0][0] = -20
                        block[-1].bias[0] = 10
                    else:
                        # Any negative value will turn to 0 when going through the ReLU
                        block[-1].weight[0][0] = -2
                        block[-1].bias[0] = 1


                # It is impossible to output a 0 through the sigmoid if the ReLU appears before it.
                # Skip it on the last step.
                if not (last_step and use_sigmoid):
                    block.append(activation())

                # Now create the block with all of the new components
                self.net.append(torch.nn.Sequential(*block))
            if use_sigmoid:
                self.net.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class BetterNet(Net):
    """Network with normalization as is normally seen in modern DNNs."""

    def __init__(self, num_steps=1, m_factor=1, activation=nn.ReLU, weight_init=True, normalize=True):
        """Initialize the demonstration network with a single output.

        Arguments:
            num_steps              (int): There will be one additional 3x3 convolution per step.
            m_factor (int): Multiplication factor. Multiply number of feature maps by this amount.
            activation (constructor): The class constructor for the activation function to use.
            weight_init (bool):
            normalize (bool):
        """
        # Initialize the parent model.
        super(BetterNet, self).__init__(num_steps=num_steps, m_factor=m_factor, presolve=False,
                activation=activation, use_sigmoid=True)
        # Now adjust weights or add normalization layers.
        with torch.no_grad():
            for i, block in enumerate(self.net):
                if type(block) is torch.nn.Sequential:
                    for j, layer in enumerate(block):
                        if hasattr(layer, 'weight'):
                            # Small variation of weights about 0, big variation of bias values.
                            nn.init.normal_(layer.weight, std=0.02)
                            nn.init.uniform_(layer.bias, a=-1.0, b=1.0)
                    # If you use uniform weights then the error can be really large! BatchNorm will
                    # prevent that from happening.
                    if i + 2 < len(self.net):
                        # Python 1.11 and newer support appending to a Sequential module, but we can
                        # do this a more awkward way for backwards compatibility. E.g.
                        #block.append(new layer)
                        # Instead recreate the layer using the block's children.
                        child_list = list(block.children())
                        #norm = torch.nn.BatchNorm2d(num_features=block[0].weight.size(1), affine=False)
                        dropout = torch.nn.Dropout2d(p=0.4)
                        # We could add a norm layer, or a norm layer and dropout, but dropout alone
                        # after the larger 3x3 convolution layer seems good.
                        #self.net[i] = torch.nn.Sequential(*child_list, norm)
                        #self.net[i] = torch.nn.Sequential(*child_list, norm, dropout)
                        self.net[i] = torch.nn.Sequential(child_list[0], dropout, *child_list[1:])
                        #self.net[i] = torch.nn.Sequential(child_list[0], dropout, *child_list[1:], norm)

        def forward(self, x):
            # Inputs are remapped to the range -0.5 to 0.5
            x = x - 0.5
            # Outputs are remapped to the same range.
            return super(BetterNet, self).forward(x) + 0.5


class DataGenerator():
    """Generate game of life data."""
    def __init__(self):
        self.conv_1_weights = torch.tensor([[[[0.,1.,0.], [1.,0.5,1.], [0.,1.,0.]]],
                                            [[[0.,1.,0.], [1.,0.5,1.], [0.,1.,0.]]]])
        self.bias_1 = torch.tensor([-2, -3.5])
        self.conv_2_weights = torch.tensor([[[[200.]], [[-800.]]]])
        self.bias_2 = torch.tensor([-10.])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def getBatch(self, batch_size, dimensions, steps):
        """Make inputs and outputs for a single output model.

        Arguments:

            batch_size (int)                : Number of elements in a batch.
            dimensions     (tuple(int, int)): Size of the input.
            steps                      (int): Number of steps from the batch examples to the output.
        Returns:
            tuple(tensor, tensor): Tuple of examples and their next states.
        """
        with torch.no_grad():
            batch = torch.randint(low=0, high=2, size=(batch_size, 1, *dimensions)).float()
            output = batch

            for _ in range(steps):

                output = self.relu(nn.functional.conv2d(
                    input=output, weight=self.conv_1_weights, bias=self.bias_1, stride=1, padding=1))

                output = torch.round(self.sigmoid(nn.functional.conv2d(
                    input=output, weight=self.conv_2_weights, bias=self.bias_2, stride=1, padding=0)))

        return batch, output


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
    if use_cuda:
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

# Show some results with the final network.
batch, labels = datagen.getBatch(batch_size=10, dimensions=(5, 5), steps=args.steps)
net = net.eval()
if use_cuda:
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
if use_cuda:
    outputs = net(batch.cuda()).round()
    labels = labels.cuda()
else:
    outputs = net(batch).round()
matches = 0
for idx in range(1000):
    if outputs[idx].equal(labels[idx]):
        matches += 1
print(f"Success rate: {matches/1000.}")
if 1000 == matches:
    print("Training success.")
else:
    print("Training failure.")

exit()
