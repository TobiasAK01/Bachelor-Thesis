import torch
from torch import nn

class JacobianRegularizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, labels, imgs):
        torch.cuda.empty_cache()
        imgs.requires_grad_()
        pred = self.model(imgs)

        grad_outputs = torch.ones_like(pred, device=pred.device)
        gradient = torch.autograd.grad(outputs=pred, inputs=imgs, grad_outputs=grad_outputs, create_graph=True)[0]

        # Compute squared sum of gradients for each example in the batch
        gradient_squared_sum = torch.sum(gradient ** 2, dim=(1, 2, 3))  # Assuming imgs is 4D tensor

        epsilon = 1e-9  # or smaller, e.g., 1e-10
        gradient_regularization = -torch.sqrt(torch.mean(torch.abs(gradient_squared_sum) + epsilon))

        return gradient_regularization