import numpy as np
import torch
import torchattacks
import torchmetrics
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image


def evaluate_adv_attack(model, dataloader, attack_name, eps, device):
    torch.cuda.empty_cache()

    ece = torchmetrics.classification.MulticlassCalibrationError(num_classes=15).to(device)
    acc = torchmetrics.classification.MulticlassAccuracy(num_classes=15).to(device)

    if hasattr(model, "models"):
        for estimator in model.models:
            estimator.eval()

    if attack_name == "FGSM":
        attack = torchattacks.FGSM(model, eps=eps)
    elif attack_name == "PGD":
        attack = torchattacks.PGD(model, eps=eps)
    elif attack_name == "BIM":
        attack = torchattacks.BIM(model, eps=eps)
    else:
        raise Exception("Given Attack not implemented.")

    attack.set_device(device)

    for imgs, labels in dataloader:
        # Move data to DEVICE
        torch.cuda.empty_cache()
        imgs, labels = imgs.to(device), labels.to(device)

        # compute Perputation on imgs
        imgs = computePerputation(imgs, labels, attack)

        # Predict
        with torch.inference_mode():
            pred = model(imgs)
            ece(pred, labels)
            acc(pred, labels)

    torch.cuda.empty_cache()
    return acc.compute().item(), ece.compute().item()


def adv_attack_hard(model, dataloader, attack_name, eps, device):
    torch.cuda.empty_cache()

    ece = torchmetrics.classification.MulticlassCalibrationError(num_classes=15).to(device)
    acc = torchmetrics.classification.MulticlassAccuracy(num_classes=15).to(device)
    model.eval()

    attacks = []
    if attack_name == "FGSM":
        for estimator in model.models:
            attack = torchattacks.FGSM(estimator, eps=eps)
            attack.set_device(device)
            attacks.append(attack)
    elif attack_name == "PGD":
        for estimator in model.models:
            attack = torchattacks.PGD(estimator, eps=eps)
            attack.set_device(device)
            attacks.append(attack)
    elif attack_name == "BIM":
        for estimator in model.models:
            attack = torchattacks.BIM(estimator, eps=eps)
            attack.set_device(device)
            attacks.append(attack)
    else:
        raise Exception("Given Attack not implemented.")

    for imgs, labels in dataloader:
        torch.cuda.empty_cache()
        # Move data to DEVICE
        imgs, labels = imgs.to(device), labels.to(device)

        # get vote
        model(imgs)
        vote = model.vote

        perb_imgs = torch.empty(imgs.size()).to(device)
        for i in range(vote.size(dim=0)):
            vote_total = vote[i]
            vote_model = []
            for j in range(model.argmax.size(dim=0)):
                if model.argmax[j][i] == vote_total:
                    vote_model.append(j)

            if vote_model.__len__() == 3:
                perb_imgs[i] = ((computePerputation(attack=attacks[vote_model[0]], imgs=imgs[i].unsqueeze(dim=0),
                                                    labels=labels[i].unsqueeze(dim=0))
                                 + computePerputation(attack=attacks[vote_model[1]], imgs=imgs[i].unsqueeze(dim=0),
                                                      labels=labels[i].unsqueeze(dim=0)))) / 2
            else:
                perb_imgs[i] = computePerputation(attack=attacks[vote_model[0]], imgs=imgs[i].unsqueeze(dim=0),
                                                  labels=labels[i].unsqueeze(dim=0))

        # Predict
        with torch.inference_mode():
            pred = model(perb_imgs)
            ece(pred, labels)
            acc(pred, labels)

    torch.cuda.empty_cache()
    return acc.compute().item(), ece.compute().item()


def adv_attack_hard_total(model, dataloader, attack_name, eps, device):
    torch.cuda.empty_cache()

    ece = torchmetrics.classification.MulticlassCalibrationError(num_classes=15).to(device)
    acc = torchmetrics.classification.MulticlassAccuracy(num_classes=15).to(device)
    if hasattr(model, "models"):
        for estimator in model.models:
            estimator.eval()

    temp_models = [model.models[0], model.models[1], model.models[2],
                   Soft_Hard(models=[model.models[0], model.models[1]]).eval()]
    attacks = []
    if attack_name == "FGSM":
        for temp_model in temp_models:
            attack = torchattacks.FGSM(temp_model, eps=eps)
            attack.set_device(device)
            attacks.append(attack)
    elif attack_name == "PGD":
        for temp_model in temp_models:
            attack = torchattacks.PGD(temp_model, eps=eps)
            attack.set_device(device)
            attacks.append(attack)
    elif attack_name == "BIM":
        for temp_model in temp_models:
            attack = torchattacks.BIM(temp_model, eps=eps)
            attack.set_device(device)
            attacks.append(attack)

    for imgs, labels in dataloader:
        torch.cuda.empty_cache()
        # Move data to DEVICE
        imgs, labels = imgs.to(device), labels.to(device)
        perb_imgs = torch.empty(imgs.size()).to(device)

        # get vote
        model(imgs)
        vote = model.vote

        # find vote per image
        for i in range(vote.size(dim=0)):
            predicted_class = vote[i]
            vote_model = []
            for j in range(model.argmax.size(dim=0)):
                # if predicted class of model == predicted class of ensemble model
                if model.argmax[j][i] == predicted_class:
                    vote_model.append(j)

            # only one model voted in favor for majority class
            if vote_model.__len__() == 3:
                perb_imgs[i] = computePerputation(attack=attacks[3], imgs=imgs[i].unsqueeze(dim=0),
                                                  labels=labels[i].unsqueeze(dim=0))

            else:
                perb_imgs[i] = computePerputation(attack=attacks[vote_model[0]], imgs=imgs[i].unsqueeze(dim=0),
                                                  labels=labels[i].unsqueeze(dim=0))

        # Predict
        with torch.inference_mode():
            pred = model(perb_imgs)
            ece(pred, labels)
            acc(pred, labels)

    torch.cuda.empty_cache()
    return acc.compute().item(), ece.compute().item()


def computePerputation(imgs, labels, attack):
    torch.cuda.empty_cache()

    return attack(imgs, labels)


class Soft_Hard(nn.Module):

    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        outputs = [
            F.softmax(model(x), dim=1) for model in self.models
        ]

        return self.average(outputs)

    def average(self, outputs):
        """Compute the average over a list of tensors with the same size."""
        return sum(outputs) / len(outputs)


def compute_adv_noise_img(model, dataloader, attack_names, eps, device, voting_strategy):
    torch.cuda.empty_cache()

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        for attack_name in attack_names:
            if voting_strategy == "hard":
                adv_attack_hard_total_imgs(model, imgs, labels, attack_name, eps, device)
                adv_attack_hard_imgs(model, imgs, labels, attack_name, eps, device)
            else:
                evaluate_adv_attack_imgs(model, imgs, labels, attack_name, eps, device)

        return


def evaluate_adv_attack_imgs(model, imgs, labels, attack_name, eps, device):
    if attack_name == "FGSM":
        attack = torchattacks.FGSM(model, eps=eps)
    elif attack_name == "PGD":
        attack = torchattacks.PGD(model, eps=eps)
    elif attack_name == "BIM":
        attack = torchattacks.BIM(model, eps=eps)
    else:
        raise Exception("Given Attack not implemented.")

    attack.set_device(device)

    # Compute perturbation on imgs
    per_img = computePerputation(attack=attack, imgs=imgs[0].unsqueeze(dim=0), labels=labels[0].unsqueeze(dim=0))

    # Compute adversarial noise
    noise = imgs[0] - per_img[0]

    # Move noise tensor to CPU and convert to NumPy array
    noise = noise.cpu().detach().numpy()

    # Convert the tensor to a NumPy array
    noise = np.transpose((noise * 255).astype(np.uint8), (1, 2, 0))
    noise = Image.fromarray(noise, 'RGB')

    # Save original, perturbed, and noise images
    save_image(imgs[0], str(eps) + attack_name + "_original.png")
    save_image(per_img, str(eps) + attack_name + "_perturbed.png")

    # Save the image if needed
    noise.save(str(eps) + attack_name + "_noise.png")


def adv_attack_hard_imgs(model, imgs, labels, attack_name, eps, device):
    model.eval()
    attacks = []
    if attack_name == "FGSM":
        for estimator in model.models:
            attack = torchattacks.FGSM(estimator, eps=eps)
            attack.set_device(device)
            attacks.append(attack)
    elif attack_name == "PGD":
        for estimator in model.models:
            attack = torchattacks.PGD(estimator, eps=eps)
            attack.set_device(device)
            attacks.append(attack)
    elif attack_name == "BIM":
        for estimator in model.models:
            attack = torchattacks.BIM(estimator, eps=eps)
            attack.set_device(device)
            attacks.append(attack)
    else:
        raise Exception("Given Attack not implemented.")

    # get vote
    model(imgs)
    vote = model.vote

    perb_imgs = torch.empty(imgs.size()).to(device)
    for i in range(vote.size(dim=0)):
        vote_total = vote[i]
        vote_model = []
        for j in range(model.argmax.size(dim=0)):
            if model.argmax[j][i] == vote_total:
                vote_model.append(j)

        if vote_model.__len__() == 3:
            perb_imgs[i] = ((computePerputation(attack=attacks[vote_model[0]], imgs=imgs[i].unsqueeze(dim=0),
                                                labels=labels[i].unsqueeze(dim=0))
                             + computePerputation(attack=attacks[vote_model[1]], imgs=imgs[i].unsqueeze(dim=0),
                                                  labels=labels[i].unsqueeze(dim=0)))) / 2
        else:
            perb_imgs[i] = computePerputation(attack=attacks[vote_model[0]], imgs=imgs[i].unsqueeze(dim=0),
                                              labels=labels[i].unsqueeze(dim=0))
        # Compute adversarial noise
        noise = imgs[0] - perb_imgs[0]

        # Move noise tensor to CPU and convert to NumPy array
        noise = noise.cpu().detach().numpy()

        # Convert the tensor to a NumPy array
        noise = np.transpose((noise * 255).astype(np.uint8), (1, 2, 0))
        noise = Image.fromarray(noise, 'RGB')

        # Save original, perturbed, and noise images
        save_image(imgs[0], "Averaged_Hard_Attack_" + str(eps) + attack_name + "_original.png")
        save_image(perb_imgs[0], "Averaged_Hard_Attack_" + str(eps) + attack_name + "_perturbed.png")

        # Save the image if needed
        noise.save("Averaged_Hard_Attack_" + str(eps) + attack_name + "_noise.png")
        return


def adv_attack_hard_total_imgs(model, imgs, labels, attack_name, eps, device):
    if hasattr(model, "models"):
        for estimator in model.models:
            estimator.eval()

    temp_models = [model.models[0], model.models[1], model.models[2],
                   Soft_Hard(models=[model.models[0], model.models[1]]).eval()]
    attacks = []
    if attack_name == "FGSM":
        for temp_model in temp_models:
            attack = torchattacks.FGSM(temp_model, eps=eps)
            attack.set_device(device)
            attacks.append(attack)
    elif attack_name == "PGD":
        for temp_model in temp_models:
            attack = torchattacks.PGD(temp_model, eps=eps)
            attack.set_device(device)
            attacks.append(attack)
    elif attack_name == "BIM":
        for temp_model in temp_models:
            attack = torchattacks.BIM(temp_model, eps=eps)
            attack.set_device(device)
            attacks.append(attack)

    perb_imgs = torch.empty(imgs.size()).to(device)

    # get vote
    model(imgs)
    vote = model.vote

    # find vote per image
    for i in range(vote.size(dim=0)):
        predicted_class = vote[i]
        vote_model = []
        for j in range(model.argmax.size(dim=0)):
            # if predicted class of model == predicted class of ensemble model
            if model.argmax[j][i] == predicted_class:
                vote_model.append(j)

        # only one model voted in favor for majority class
        if vote_model.__len__() == 3:
            perb_imgs[i] = computePerputation(attack=attacks[3], imgs=imgs[i].unsqueeze(dim=0),
                                              labels=labels[i].unsqueeze(dim=0))

        else:
            perb_imgs[i] = computePerputation(attack=attacks[vote_model[0]], imgs=imgs[i].unsqueeze(dim=0),
                                              labels=labels[i].unsqueeze(dim=0))
        # Compute adversarial noise
        noise = imgs[0] - perb_imgs[0]

        # Move noise tensor to CPU and convert to NumPy array
        noise = noise.cpu().detach().numpy()

        # Convert the tensor to a NumPy array
        noise = np.transpose((noise * 255).astype(np.uint8), (1, 2, 0))
        noise = Image.fromarray(noise, 'RGB')

        # Save original, perturbed, and noise images
        save_image(imgs[0], "Soft_Hard_Attack_" + str(eps) + attack_name + "_original.png")
        save_image(perb_imgs[0], "Soft_Hard_Attack_" + str(eps) + attack_name + "_perturbed.png")

        # Save the image if needed
        noise.save("Soft_Hard_Attack_" + str(eps) + attack_name + "_noise.png")
        return
