import torch
from torch import nn
import torch.nn.functional as f

class ChiSquareDistance(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, labels, imgs):
        torch.cuda.empty_cache()
        outputs = [f.softmax(estimator(imgs), dim=1) for estimator in self.model.models]

        M = len(outputs)
        factor = 1 / (M * (M - 1))
        pairwise_distance = []
        for i in range(M):
            for j in range(M):
                if i != j:
                    numerator = torch.pow(outputs[i] - outputs[j], exponent=2)
                    denominator = outputs[i] + outputs[j]
                    pairwise_distance.append(torch.div(numerator, denominator))

        res_temp = torch.mul(factor, torch.stack(pairwise_distance, dim=0).sum(dim=0))
        return torch.mean(torch.log(res_temp))