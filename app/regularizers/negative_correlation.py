class NegativeCorrelation(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, labels, imgs):
        torch.cuda.empty_cache()

        pred = self.model(imgs)
        outputs = [F.softmax(estimator(imgs), dim=1) for estimator in self.model.models]

        neg_corr = 0
        for i, output_i in enumerate(outputs):
            diff_i = output_i - pred
            sum_j = sum(outputs[j] - pred for j in range(len(outputs)) if j != i)
            neg_corr -= torch.sum(torch.mul(diff_i, sum_j))

        return neg_corr / len(outputs)