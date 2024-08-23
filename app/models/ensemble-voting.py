import torch
from torch import nn
import torch.nn.functional as F


class EnsembleVoting(nn.Module):
    def __init__(self, models, voting_strategy):
        super().__init__()
        implemented_strategies = {"soft", "hard", "weighted_acc", "weighted_robust"}
        if voting_strategy not in implemented_strategies:
            msg = (
                "Voting strategy {} is not implemented, "
                "please choose from {}."
            )
            raise ValueError(
                msg.format(voting_strategy, implemented_strategies)
            )

        self.voting_strategy = voting_strategy
        self.models = models

        self.weights = []
        for _ in self.models:
            self.weights.append(1)

        self.vote = None
        self.argmax = None
        self.max = None

    def forward(self, x):
        outputs = [
            models(x) for models in self.models
        ]

        if self.voting_strategy == "soft":
            proba = self.average(outputs)
        elif self.voting_strategy == "weighted_acc":
            proba = self.weighted_average_acc(outputs)
        elif self.voting_strategy == "weighted_robust":
            proba = self.weighted_average_robust(outputs)
        else:
            proba = self.majority_vote(outputs)

        return proba

    def average(self, outputs):
        """Compute the average over a list of tensors with the same size."""
        return torch.div(torch.stack(outputs, dim=0).sum(dim=0), len(outputs))

    def weighted_average_acc(self, outputs):
        self.weights = F.softmax(torch.tensor(self.weights, device="cuda:"+str(outputs[0].get_device()), requires_grad=True), dim=0)
        weighted_outputs = []
        for i in range(len(outputs)):
            weighted_outputs.append(torch.mul(outputs[i], self.weights[i]))
        return torch.stack(weighted_outputs, dim=0).sum(dim=0), len(outputs)

    def weighted_average_robust(self, outputs):
        self.weights = F.softmax(torch.tensor(self.weights, device="cuda:"+str(outputs[0].get_device()), requires_grad=True, dtype=torch.float64), dim=0)
        weighted_outputs = []
        for i in range(len(outputs)):
            weighted_outputs.append(torch.mul(outputs[i], self.weights[i]))
        return torch.stack(weighted_outputs, dim=0).sum(dim=0), len(outputs)

    def majority_vote(self, outputs):
        """Compute the majority vote for a list of model outputs.
        outputs: list of length (n_models)
        containing tensors with shape (n_samples, n_classes)
        majority_one_hots: (n_samples, n_classes)
        """

        if len(outputs[0].shape) != 2:
            msg = """The shape of outputs should be a list tensors of
            length (n_models) with sizes (n_samples, n_classes).
            The first tensor had shape {} """
            raise ValueError(msg.format(outputs[0].shape))
        stack = torch.stack(outputs)
        argmax = stack.argmax(dim=2)
        self.argmax = argmax

        max = stack.max(dim=2)
        self.max = max[0]
        votes = argmax.mode(dim=0)
        self.vote = votes[0]
        proba = torch.zeros_like(outputs[0])
        majority_one_hots = proba.scatter_(1, votes[0].view(-1, 1), 1)

        return majority_one_hots

