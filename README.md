# dnn_game_of_life

Implementation of the game of life with a neural network, as described in:
[https://arxiv.org/abs/2009.01398](https://arxiv.org/abs/2009.01398)

This repository has code to train models to solve the game of life (in `gameoflife.py`) and scripts
to generate artifacts that illustrate the effectiveness of stochastic gradient descent at that task.

## Overview

The Game of Life, created by John Horton Conway, is played on a grid of tiles. Each tile is either
"alive" or "dead" (or black and white for visual purposes. It is a rough simulation of life; if a
cell is surrounded by too many other cells (4) then it "starves" and is no longer living in the next
time step. If a living cell does not have enough living neighbors (less than 2) then it also dies
(from loneliness I suppose) and empty cells can be brought to life if they have three living
neighbors. Diagonals don't count as neighbors, only the four adjacent cells.

The rules for a single cell only look at the cell and the four adjacent cells, so a 3x3 convolution
should be able to determine the next state.

The network defined in PyTorch looks something like this:

    Net(
      (net): ModuleList(
        (0): Sequential(
          (0): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU()
          (2): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
          (3): ReLU()
          (4): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (5): ReLU()
        )
      )
    )

There are two convolution kernels in the first layer to separate the two different thresholds:
* The minimum number of neighbors for a cell to be alive in the next step
* The minimum number of neighbors for a cell to be dead in the next step from overcrowding

The weights of the convolutions are `[[0, 1, 0], [1, 0.5, 1], [0, 1, 0]]`.

If we label alive a `1` and dead as `0`, then the output of that filter is:
1. cell is 0 or 1, 1 or fewer neighbors: output <= 1.5
2. cell is 0, 2 neighbors: output = 2
3. cell is 1, 2 neighbors: output = 2.5
4. cell is 0, 3 neighbors: output = 3
5. cell is 1, 3 neighbors: output = 3.5
6. cell is 0 or 1, 4 neighbors: output >= 4

The cell is alive (a 1) in the next time step only in cases 3, 4, and 5. It is dead (a 0) otherwise.

The first filter thus has a bias of -2. If the output is negative then the cell is dead.

The second filter has a bias of -3.5. If the output is positive then the cell is dead.

The ReLU after the first convolution turns all negative values to 0, so the cell is alive iff:
* The first filter output is > 0.
* The second filter output is 0.

The 1x1 convolution combines the two filter outputs (feature maps) into a single feature map. We
could simply multiply the first feature by 1 and the second with -4 (which is the maximum output of the first
feature map): if the sum is positive then the cell is alive. However, we want to get an output of
exactly 1 when the cell is alive, so we use two 1x1 filters. The first 1x1 will detect if the cell
is dead, and the second will output a 1 unless the dead indication is positive.

To accomplish that, the first 1x1 filter will apply weights of -2 and 10 to the first layer's
first and second feature maps, respectively, and a bias of 1. Recall that if the second feature map
is positive then the cell is dead, which is why we multiple by a large value: even if the output of
the second filter is 0.5 we want it to overwhelm the value of the first filter, which has an output
of 2 when the second filter is 0.5.

The second 1x1 filter has a bias of 1 so that it default to having the cell alive. A weight of -2 on
the single input feature guarantees that the output goes to 0 (after the ReLU) if the first 1x1 has
a positive output.

Thus we achieve an exact solution to the game of life with this three layer network. To obtain
networks that can successfully predict multiple steps of the game of life, we can just duplicate the
three layers for as many steps as desired.

## Training Models

Training a model is quite simple and is already set up with the command:
> python3 gameoflife.py

If you run that command you will (most likely) see it fail to create a network that successfully
outputs the next state given an initial board. You can control the starting values with the `--seed`
option, but don't bother searching around for a successful run, it is very unlikely.

The `--use_sigmoid` option will aim for an inexact solution that may be easier to reach but putting
a sigmoid activation at the end of the network. This will force any large negative value to be close
to 0 and any large positive value to be close to 1. This still won't get the network to converge to
the correct answer. In order to start seeing success, we will need to make the network larger than
necessary with the `--m_factor` and `--d_factor` options. They will, respectively, construct a
network wider and deep than required.

Running this command will have a high likelihood of creating a mostly successfull network:
> python3 gameoflife.py --seed 2 --use_sigmoid --batches 10000 --m_factor 5

## Analyzing Results

Use the script in `analysis/get_success_results.sh` to train models and find the success probability
for multiple different steps and for different values of `m_factor` and `d_factor`. The training
command uses the following options:
* `--use_sigmoid`: This makes convergence simpler by allowing "close enough" outputs
* `--activation_fun LeakyReLU`: This gives more gradients than ReLU. Works well when "close enough"
  outputs from the sigmoid are being used.
* `--normalize`: This normalizes the outputs to the range -0.5 and 0.5 (instead of 0 and 1), changes
  initial weight initialization to a small range around 0, bias is uniform from -1 to 1, and adds dropout. Typical tweaks from a professional setting.
* `--batches 25000`: Give more time to converge.
* `--batch_size 128`: An aggressive batch size helps keep learning smooth.

Get training results and put them into a text file:
> bash get_success_results.sh > success_results.dat
