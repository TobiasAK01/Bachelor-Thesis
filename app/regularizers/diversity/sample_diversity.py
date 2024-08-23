import torch
from torch import nn
import torch.nn.functional as f


class SampleDiversity(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, labels, imgs):
        torch.cuda.empty_cache()

        # white noise out of distibution datapoints
        w_datapoints = torch.rand((128, 3, 50, 50), device=self.device, requires_grad=True)
        # compute pred on white noise per model
        output = []
        for estimator in self.model.models:
            output.append(f.softmax(estimator(w_datapoints), dim=1))

        # compute logit and normalize to length one per output
        norm_logit_output = []
        for i in output:
            norm_logit_output.append(f.normalize(torch.logit(i, eps=1e-6), p=2, dim=1, eps=1e-12))

        matrix_T = torch.stack(norm_logit_output, dim=1)
        matrix = torch.transpose_copy(matrix_T, dim0=1, dim1=2)
        logdet = torch.logdet(torch.matmul(matrix_T, matrix))
        return torch.mean(logdet)