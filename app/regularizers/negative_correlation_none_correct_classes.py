class NegativeCorrelationNC(nn.Module):
    def __init__(self, device, model):
        super().__init__()
        self.device = device
        self.model = model

    def forward(self, labels, imgs):
        torch.cuda.empty_cache()

        pred = self.model(imgs)
        outputs = [F.softmax(estimator(imgs), dim=1) for estimator in self.model.models]

        # empty tensor of size batchsize x C-1 for storing y_k's
        pred_without_k = torch.empty((pred.size(dim=0), pred.size(dim=1) - 1), device=self.device)

        # list for storing empty tensors of size batchsize x C-1 for storing y_k's
        output_without_k = []
        for i in range(len(outputs)):
            output_without_k.append(torch.empty((pred.size(dim=0), pred.size(dim=1) - 1), device=self.device))

        for i in range(labels.size(dim=0)):
            # index of k (ground truth index)
            label_index = labels[i].item()

            # store y_k
            pred_without_k[i] = F.softmax(torch.cat((pred[i][0:label_index], pred[i][label_index + 1:]), dim=0), dim=0)

            for j in range(len(outputs)):
                output_without_k[j][i] = F.softmax(
                    torch.cat((outputs[j][i][0:label_index], outputs[j][i][label_index + 1:]), dim=0), dim=0)

        neg_corr = 0
        for i, output_i in enumerate(output_without_k):
            diff_i = torch.sub(output_without_k[i], pred_without_k)
            sum_j = sum(output_without_k[j] - pred_without_k for j in range(len(output_without_k)) if j != i)
            neg_corr -= torch.sum(torch.mul(diff_i, sum_j))

        return neg_corr / len(outputs)