import torch
from torch import nn
import torch.nn.functional as f

class ADP(nn.Module):

    def __init__(self, alpha, beta, device, model):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.model = model

    def forward(self, labels, imgs):
        torch.cuda.empty_cache()

        pred = self.model(imgs)
        outputs = [f.softmax(estimator(imgs), dim=1) for estimator in self.model.models]

        # Calculate pred_without_k and matrix_T
        pred_without_k, matrix_T = self.calculate_pred_without_k_and_matrix_T(pred, outputs, labels)

        # Calculate ADP regularization
        entropy_term = self.calculate_entropy(pred_without_k)
        det_term = self.calculate_determinant_term(matrix_T)

        return torch.mean(self.alpha * entropy_term + self.beta * det_term)

    def calculate_pred_without_k_and_matrix_T(self, pred, outputs, labels):
        pred_without_k = []
        matrix_T = []

        for i in range(labels.size(dim=0)):
            label_index = labels[i].item()

            # Calculate pred_without_k
            pred_without_k_i = torch.cat((pred[i][:label_index], pred[i][label_index + 1:]), dim=0)
            pred_without_k_i = f.normalize(pred_without_k_i, p=2, dim=0)
            pred_without_k.append(pred_without_k_i)

            # Calculate matrix_T
            matrix_T_i = torch.empty((len(outputs), pred.size(dim=1) - 1)).to(self.device)
            for j in range(len(outputs)):
                output_without_k = torch.cat((outputs[j][i][:label_index], outputs[j][i][label_index + 1:]), dim=0)
                output_without_k = f.normalize(output_without_k, p=2, dim=0)
                matrix_T_i[j] = output_without_k
            matrix_T.append(matrix_T_i)

        return torch.stack(pred_without_k), torch.stack(matrix_T)

    def calculate_determinant_term(self, matrix_T):
        det_terms = []
        for i in range(len(matrix_T)):
            matrix = matrix_T[i].transpose(0, 1)  # Transpose matrix_T[i]
            det = torch.det(matrix @ matrix.T)  # Calculate determinant using SVD
            det_terms.append(det)
        return torch.log(torch.stack(det_terms) + 1e-20)

    def calculate_entropy(self, pred_without_k):
        probs = f.softmax(pred_without_k, dim=1)
        log_probs = torch.log(probs + 1e-20)  # Add a small epsilon to prevent log(0)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy