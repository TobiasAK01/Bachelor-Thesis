import torch
from torch import nn
import torch.nn.functional as f


class GradientIndependence(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, labels, imgs):
        torch.cuda.empty_cache()
        imgs.requires_grad_()

        estimators_gradients = []
        for estimator in self.model.models:
            # Forward pass to compute the gradients
            pred = estimator(imgs)
            loss = F.cross_entropy(pred, labels)
            gradients = torch.autograd.grad(loss, imgs, create_graph=True)[0]

            # Flatten and normalize the gradients
            flattened_gradients = torch.flatten(gradients, start_dim=1)
            normalized_gradients = f.normalize(flattened_gradients, p=2, dim=1, eps=1e-12)
            estimators_gradients.append(normalized_gradients)

        # Stack the normalized gradients of all estimators
        matrix_T = torch.stack(estimators_gradients, dim=1)

        # Compute the transpose of the stacked matrix
        matrix = torch.transpose(matrix_T, dim0=1, dim1=2)

        # Compute the determinant of the product of the stacked matrix and its transpose
        determinant = torch.det(torch.matmul(matrix_T, matrix))

        # Compute the logarithm of the determinant and take the mean
        log_det = torch.log(determinant)
        return torch.mean(log_det)
