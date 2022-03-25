import torch
import torch.nn as nn

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

