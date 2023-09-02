import torch
import torch.nn as nn

class Net(torch.nn.Module):
    """Network that demonstrates some fundamentals of fitting."""

    def __init__(self, num_steps=1, m_factor=1, presolve=True, activation=nn.ReLU,
            use_sigmoid=False, solve_layers=[]):
        """Initialize the demonstration network with a single output.

        Arguments:
            num_steps              (int): There will be one additional 3x3 convolution per step.
            m_factor (int): Multiplication factor. Multiply number of feature maps by this amount.
            presolve (bool): Initialize weights and bias values to the solution if True.
            activation (constructor): The class constructor for the activation function to use.
            solve_layers (List[int]): Individual layers to presolve
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
                if presolve or (step in solve_layers):
                    block[-1].weight[0][0] = self.weighted_sum
                    block[-1].weight[1][0] = self.weighted_sum
                    block[-1].bias[0] = -2
                    block[-1].bias[1] = -3.5
                    # Zero any excess layers
                    block[-1].weight[2:].fill_(0.)
                    block[-1].bias[2:].fill_(0.)
                block.append(activation())

                block.append(
                    nn.Conv2d(
                        in_channels=2*m_factor, out_channels=m_factor, kernel_size=1, stride=1, padding=0))
                # The final output should end up 0 when the second layer is 4 times the first. Use a
                # ratio of 1:5 instead of 1:4 to move the output away from 0.  The next layer will
                # set an exact output 1 using bias, so if the output should be alive make it
                # negative here. A negative weight at the next layer will map a positive output from
                # this layer to 0 once it passes through a ReLU.
                if presolve or (step in solve_layers):
                    block[-1].weight[0][0] = -2
                    block[-1].weight[0][1] = 10
                    block[-1].bias[0] = 1
                    # Zero any excess layers
                    block[-1].weight[1:].fill_(0.)
                    block[-1].bias[1:].fill_(0.)
                block.append(activation())

                block.append(
                    nn.Conv2d(
                        in_channels=m_factor, out_channels=1, kernel_size=1, stride=1, padding=0))
                if presolve or (step in solve_layers):
                    if use_sigmoid and last_step:
                        # Set things up for the sigmoid
                        block[-1].weight[0][0] = -20
                        block[-1].bias[0] = 10
                    else:
                        # Any negative value will turn to 0 when going through the ReLU
                        block[-1].weight[0][0] = -2
                        block[-1].bias[0] = 1
                    # Zero any excess layers
                    block[-1].weight[1:].fill_(0.)
                    block[-1].bias[1:].fill_(0.)

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

    def __init__(self, num_steps=1, m_factor=1, activation=nn.ReLU, normalize=True):
        """Initialize the demonstration network with a single output.

        Arguments:
            num_steps              (int): There will be one additional 3x3 convolution per step.
            m_factor (int): Multiplication factor. Multiply number of feature maps by this amount.
            activation (constructor): The class constructor for the activation function to use.
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


class SinBackedConv(torch.nn.Module):
    """A 2D conv layer with weights represented by a sine function."""

    def __init__(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
            bias=True, padding_model='zeros', device=None, dtype=None):
        """Mirror the torch.nn.Conv2d constructor."""
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Use the sin function and initialize the phase from -0.5*pi to 0.5*pi
        # One sin function for all outputs at every unique c,y,x input coordinate
        # Each sin function will have a phase and a sample interval
        self.phases = torch.tensor(out_channels, kernel_size, kernel_size)
        self.intervals = torch.tensor(out_channels, kernel_size, kernel_size)
        nn.init_uniform_(self.phases, a=-0.5*math.pi, b=0.5*math.pi)
        nn.init_uniform_(self.intervals, std=math.pi)







class CompactNet(torch.nn.Module):
    """A neural network that minimizes the number of variables representing the weights of a layer."""

    def __init__(self, num_steps=1, m_factor=1, activation=nn.ReLU,
            weight_init=True, normalize=True, use_sigmoid=True, presolve=False):
        """Initialize the compact network to solve the game of life.

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
