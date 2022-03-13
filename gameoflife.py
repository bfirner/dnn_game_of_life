#! /usr/bin/python3

import argparse
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

                self.net.append(
                    nn.Conv2d(
                        in_channels=1, out_channels=2*m_factor, kernel_size=3, stride=1, padding=1))
                if presolve:
                    self.net[-1].weight[0][0] = self.weighted_sum
                    self.net[-1].weight[1][0] = self.weighted_sum
                    self.net[-1].bias[0] = -2
                    self.net[-1].bias[1] = -3.5
                self.net.append(activation())

                self.net.append(
                    nn.Conv2d(
                        in_channels=2*m_factor, out_channels=m_factor, kernel_size=1, stride=1, padding=0))
                # The final output should end up 0 when the second layer is 4 times the first. Use a
                # ratio of 1:5 instead of 1:4 to move the output away from 0.  The next layer will
                # set an exact output 1 using bias, so if the output should be alive make it
                # negative here. A negative weight at the next layer will map a positive output from
                # this layer to 0 once it passes through a ReLU.
                if presolve:
                    self.net[-1].weight[0][0] = -2
                    self.net[-1].weight[0][1] = 10
                    self.net[-1].bias[0] = 1
                self.net.append(activation())

                self.net.append(
                    nn.Conv2d(
                        in_channels=m_factor, out_channels=1, kernel_size=1, stride=1, padding=0))
                if presolve:
                    if use_sigmoid and last_step:
                        # Set things up for the sigmoid
                        self.net[-1].weight[0][0] = -20
                        self.net[-1].bias[0] = 10
                    else:
                        # Any negative value will turn to 0 when going through the ReLU
                        self.net[-1].weight[0][0] = -2
                        self.net[-1].bias[0] = 1


                # It is impossible to output a 0 through the sigmoid if the ReLU appears before it.
                # Skip it on the last step.
                if not (last_step and use_sigmoid):
                    self.net.append(activation())
            if use_sigmoid:
                self.net.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

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
    help='Overprovisioning factor for the neural network.')
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
    '--activation_fun',
    required=False,
    default='ReLU',
    choices=['ReLU', 'ELU', 'LeakyReLU', 'RReLU', 'GELU'],
    type=str,
    help="Nonlinear activation function to use in the network.")
args = inparser.parse_args()

afun = getattr(torch.nn, args.activation_fun)
net = Net(num_steps=args.steps, m_factor=args.m_factor, presolve=args.presolve, activation=afun,
        use_sigmoid=args.use_sigmoid)
optimizer = torch.optim.Adam(net.parameters())
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


for batch_num in range(args.batches):
    batch, labels = datagen.getBatch(batch_size=args.batch_size, dimensions=(10, 10), steps=args.steps)
    optimizer.zero_grad()
    out = net.forward(batch)
    loss = loss_fn(out, labels)
    if 0 == batch_num % 1000:
        with torch.no_grad():
            print(f"Batch {batch_num} loss is {loss.mean()}")
            # This is a bit too spammy with larger networks.
            if 1 == args.steps:
                for i, layer in enumerate(net.net):
                    if hasattr(layer, 'weight'):
                        print(f"At batch {batch_num} layer {i} has weights {layer.weight.tolist()} and bias {layer.bias.tolist()}")
    loss.backward()
    optimizer.step()


print("Final layer weights are:")
for i, layer in enumerate(net.net):
    if hasattr(layer, 'weight'):
        print(f"Final layer {i} weights are {layer.weight.tolist()} and bias is {layer.bias.tolist()}")

# Show some results with the final network.
batch, labels = datagen.getBatch(batch_size=10, dimensions=(3, 3), steps=args.steps)
net = net.eval()
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
exit()
