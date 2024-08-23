class SampleDiverstiyAdversarialNoise(nn.Module):

    def __init__(self, model, eps, device):
        super().__init__()
        self.model = model
        self.eps = eps
        self.attack = torchattacks.FGSM(self.model, eps=self.eps)
        self.device = device
        self.attack.set_device(device)

    def forward(self, labels, imgs):
        torch.cuda.empty_cache()

        # compute adv_imgs
        adv_imgs = computePerputation(imgs, labels, self.attack)

        # adv noise
        adv_noise = torch.sub(adv_imgs, imgs)

        # compute pred on adv noise per model
        output = []
        for estimator in self.model.models:
            output.append(F.softmax(estimator(adv_noise), dim=1))

        # compute logit and normalize to length one per output
        norm_logit_output = []
        for i in output:
            norm_logit_output.append(F.normalize(torch.logit(i, eps=1e-6), p=2, dim=1, eps=1e-12))

        matrix_T = torch.stack(norm_logit_output, dim=1)
        matrix = torch.transpose_copy(matrix_T, dim0=1, dim1=2)
        logdet = torch.logdet(torch.matmul(matrix_T, matrix))
        return torch.mean(logdet)