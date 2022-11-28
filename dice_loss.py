import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    This module implements the Dice loss function used to train the model
    """
    def __init__(self, weights=None) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.weights = weights

    def forward(self, input: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Method used to compute the Dice Loss function
        :param input: the input is the probabilities for each mask of shape (n_classes, height, width)
        :param target_one_hot: output masks of shape (n_classes, height, width)
        :return: the computed value of the Dice Loss
        """
        dims = (1, 2, 3)

        # if self.weights is not None:
        #     for i in range(len(self.weights)):
        #         input[i] = input[i] * self.weights[i]

        intersection = torch.sum(input * target_one_hot, (2,3))

        cardinality = torch.sum(input + target_one_hot, (2,3))
        dice_score = 2. * intersection / (cardinality + self.eps)

        loss = torch.mean(1. - dice_score, 0)

        return torch.dot(loss.float(), self.weights.float())