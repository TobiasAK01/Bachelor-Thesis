import torch
from torch import nn

class GradientRegularizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, labels, imgs):
        torch.cuda.empty_cache()
        imgs.requires_grad_()
        pred = self.model(imgs)

        # Compute cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        ce_loss = criterion(pred, labels)

        # Compute gradient of cross-entropy loss w.r.t. input images
        grad_outputs = torch.ones_like(ce_loss)
        gradient = torch.autograd.grad(outputs=ce_loss, inputs=imgs, grad_outputs=grad_outputs, create_graph=True)[0]

        # Compute squared sum of gradients for each example in the batch
        gradient_squared_sum = torch.sum(gradient ** 2, dim=(1, 2, 3))  # Assuming imgs is 4D tensor

        # Gradient regularization term is the mean of squared sum of gradients
        gradient_regularization = -torch.mean(gradient_squared_sum)

        return gradient_regularization