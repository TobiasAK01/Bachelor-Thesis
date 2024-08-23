import torch
import torchmetrics
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import torch.nn

from Data.dataset import CustomDataset
from Attack.attacks import evaluate_adv_attack, adv_attack_hard, \
    adv_attack_hard_total, compute_adv_noise_img
from Models.EnsembleVoting import EnsembleVoting
from utils import writeResultsEpoch, getModel, getDataLoaders, getOptimizer, writeTestResults, configure_regularizer, \
    visualizeFeatures, visualize_gradient, visualize_feature_distribution, visualize_feature_distribution_hard

# Hyperparameters für Dataloader
BATCH_SIZE = 256
NUM_WORKERS = 7
DEVICE = "cuda:0"
K_FOLDS = 2
# Set fixed random number seed
torch.manual_seed(42)

# Hyperparameters for Model and Training
MODEL = "resnet"  # resnet, densnet, shufflenet, ensemble
VOTING_STRATEGY = ""  # soft, hard, weighted_acc, weighted_robust, ""
LR = 0.0005
WEIGHT_DECAY = 0.0002
EPOCHS = 1

# Hyperparameters for Regularizer
REGULARIZER = ""  # NC, NC_NCORRECT, SD, SD_ADV, ADP, CHI, GI, DGR, DJR, IJR, IGR
LAMBDA = 0.5
ALPHA = 0.125
BETA = 0.5

# Save/Load Path
WRITER = "runs/" + MODEL + "/" + VOTING_STRATEGY + "/" + REGULARIZER
ADV_IMG = True
HIDDEN_FEATURES = False
TEST = False
GRADIENT = False
FEATURE_DISTRIBUTION = False

# Hyperparameter for adversarial Attacks
ATTACKS = ["FGSM", "PGD", "BIM"]
EPS = [0.01, 0.05, 0.1]


def inittrain():
    writer = SummaryWriter(WRITER)
    dataset = CustomDataset()
    kfold = KFold(n_splits=K_FOLDS, shuffle=True)
    metric_dic = {}

    for fold, (train_val_ids, test_ids) in enumerate(kfold.split(dataset)):
        print('--------------------------------')
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Init DataLoader, Model, Regularization and Optimizers
        dataloader_train, dataloader_val, dataloader_test = getDataLoaders(train_val_ids, test_ids,
                                                                           numworkers=NUM_WORKERS,
                                                                           batchsize=BATCH_SIZE)
        model = getModel(model=MODEL, voting_strategy=VOTING_STRATEGY, device=DEVICE)
        optimizers = getOptimizer(modelInstance=model, weight_decay=WEIGHT_DECAY, lr=LR, model=MODEL)
        regularizer = configure_regularizer(model, REGULARIZER, DEVICE, ALPHA, BETA, 0.1, VOTING_STRATEGY)

        # ------------------------Train Loop------------------------ #
        for epoch in range(EPOCHS):
            # Print epoch
            print(f'Starting epoch {epoch + 1}')
            print('--------------------------------')

            # ------------------------Train one Epoch------------------------ #
            train_loss = train(model, dataloader_train, optimizers, regularizer, DEVICE, REGULARIZER, LAMBDA)

            # ------------------------Calculate Weights if necessary------------------------ #
            if VOTING_STRATEGY == "weighted_acc" or VOTING_STRATEGY == "weighted_robust":
                calculate_weights(model, dataloader_val, 0.1, DEVICE, VOTING_STRATEGY)

            # ------------------------Validate------------------------ #
            val_acc, val_loss, ece = evaluate(model, dataloader_val, DEVICE)

            # ------------------------Write Train-Loss, Val-Loss, Val-Acc------------------------ #
            writeResultsEpoch(writer, epoch, train_loss, "train/loss")
            writeResultsEpoch(writer, epoch, val_loss, "val/loss")
            writeResultsEpoch(writer, epoch, val_acc * 100, "val/acc")
            writeResultsEpoch(writer, epoch, ece * 100, "val/ece")

        # ------------------------Testing, Adversarial Attacks------------------------ #
        if TEST:
            test(metric_dic, model, dataloader_test, VOTING_STRATEGY, DEVICE, ATTACKS, EPS)

        if ADV_IMG:
            compute_adv_noise_img(model, dataloader_test, ATTACKS, 0.05, DEVICE, VOTING_STRATEGY)

        if HIDDEN_FEATURES:
            visualizeFeatures(ensemble_model=model, device=DEVICE, dataloader=dataloader_test, model=MODEL, voting_strategy=VOTING_STRATEGY, regularizer=REGULARIZER)

        if GRADIENT:
            visualize_gradient(ensemble_model=model, dataloader=dataloader_test, device=DEVICE, model=MODEL, voting_strategy=VOTING_STRATEGY, regularizer=REGULARIZER)

        if FEATURE_DISTRIBUTION:
            if VOTING_STRATEGY == "hard":
                visualize_feature_distribution_hard(ensemble_model=model, dataloader=dataloader_test, device=DEVICE, model=MODEL, voting_strategy=VOTING_STRATEGY, regularizer=REGULARIZER)
            else:
                visualize_feature_distribution(ensemble_model=model, dataloader=dataloader_test, device=DEVICE, model=MODEL, voting_strategy=VOTING_STRATEGY, regularizer=REGULARIZER)

        break


    writeTestResults(writer, {"Model": MODEL, "Regularizer": REGULARIZER, "Voting Strategy": VOTING_STRATEGY},
                     metric_dic)


def train(model, dataloader, optimizers, regularizer, device, regularizer_name, hlambda):
    if regularizer_name in {"NC", "NC_NCORRECT", "SD", "SD_ADV", "ADP", "CHI", "GI", "DGR",
                            "DJR"} and MODEL == "ensemble":
        train_loss = train_depend(model, dataloader, optimizers, regularizer, device, hlambda)
    else:
        models = []
        if MODEL == "ensemble":
            models = model.models
        else:
            models.append(model)
        train_loss = train_independent(models, dataloader, optimizers, regularizer, device, hlambda)

    return train_loss


def train_depend(model, dataloader, optimizers, regularizer, device, hlambda):
    torch.cuda.empty_cache()

    train_loss = 0

    model.train()
    for imgs, labels in dataloader:
        # Move data to device
        imgs, labels = imgs.to(device), labels.to(device)

        losses = []
        for i in range(len(model.models)):
            # Zero your gradients for every batch
            optimizers[i].zero_grad()

            # Compute the loss, its gradient and save for visualization
            losses.append(F.cross_entropy(model.models[i](imgs), labels))

        loss = torch.stack(losses, dim=0).sum(dim=0)
        if regularizer:
            reg = regularizer(labels=labels, imgs=imgs)
            loss -= hlambda * reg
        train_loss += loss.item()
        loss.backward()

        for i in range(len(model.models)):
            optimizers[i].step()

    return train_loss / len(dataloader.dataset)


def train_independent(models, dataloader, optimizers, regularizer, device, hlambda):
    torch.cuda.empty_cache()

    train_loss = 0
    for i in range(len(models)):
        models[i].train()
        for imgs, labels in dataloader:
            # Move data to device
            imgs, labels = imgs.to(device), labels.to(device)

            # Zero your gradients for every batch
            optimizers[i].zero_grad()

            # Predict
            pred = models[i](imgs)

            # Compute the loss, its gradient and save for visualization
            loss = F.cross_entropy(pred, labels)
            if regularizer:
                reg = regularizer[i](labels=labels, imgs=imgs)
                loss -= hlambda * reg

            train_loss += loss.item()
            loss.backward()

            # Adjust learning weights
            optimizers[i].step()

    return train_loss / len(dataloader.dataset)


def test(metric_dic, model, dataloader, voting_strategy, device, attacks, eps_list):
    print('Starting testing')
    test_acc, _, ece = evaluate(model, dataloader, device)

    if "Accuracy" in metric_dic:
        metric_dic["Accuracy"] = (metric_dic.get("Accuracy") + test_acc * 100) / 2
    else:
        metric_dic["Accuracy"] = test_acc * 100

    if "ECE" in metric_dic:
        metric_dic["ECE"] = (metric_dic.get("ECE") + ece * 100) / 2
    else:
        metric_dic["ECE"] = ece * 100

    print(f'Accuracy: {test_acc * 100}')
    print(f'ECE: {ece * 100}')

    fro = calculateFrobeniusGradient(model, dataloader, device, voting_strategy)
    print(f'Frobenius Norm: {fro}')

    if "Frobenius Norm" in metric_dic:
        metric_dic["Frobenius Norm"] = (metric_dic.get("Frobenius Norm") + fro) / 2
    else:
        metric_dic["Frobenius Norm"] = fro

    print('Starting Attacks')
    for attack in attacks:
        for eps in eps_list:

            if voting_strategy != "hard":
                acc1, ece1 = evaluate_adv_attack(model, dataloader, attack, eps, device)
                print("Total Accuracy " + attack + " with " + str(eps) + ": " + str(acc1 * 100))
                print("Total ECE " + attack + " with " + str(eps) + ": " + str(ece1 * 100))

                if "Total Accuracy " + attack + ": " + str(eps) in metric_dic:
                    metric_dic["Total Accuracy " + attack + ": " + str(eps)] = (metric_dic.get(
                        "Total Accuracy " + attack + ": " + str(eps)) + acc1 * 100) / 2
                    metric_dic["Total ECE " + attack + ": " + str(eps)] = (metric_dic.get(
                        "Total ECE " + attack + ": " + str(eps)) + ece1 * 100) / 2
                else:
                    metric_dic["Total Accuracy " + attack + ": " + str(eps)] = acc1 * 100
                    metric_dic["Total ECE " + attack + ": " + str(eps)] = ece1 * 100

            else:
                acc1, ece1 = adv_attack_hard(model, dataloader, eps=eps, attack_name=attack, device=device)
                print("Accumulated Accuracy " + attack + " with " + str(eps) + ": " + str(acc1 * 100))
                print("Accumulated ECE " + attack + " with " + str(eps) + ": " + str(ece1 * 100))

                acc2, ece2 = adv_attack_hard_total(model, dataloader, eps=eps, attack_name=attack, device=device)
                print("Total Accuracy " + attack + " with " + str(eps) + ": " + str(acc2 * 100))
                print("Total ECE " + attack + " with " + str(eps) + ": " + str(ece2 * 100))

                if "Accumulated Accuracy " + attack + ": " + str(eps) in metric_dic:
                    metric_dic["Accumulated Accuracy " + attack + ": " + str(eps)] = (metric_dic.get(
                        "Accumulated Accuracy " + attack + ": " + str(eps)) + acc1 * 100) / 2
                    metric_dic["Accumulated ECE " + attack + ": " + str(eps)] = (metric_dic.get(
                        "Accumulated ECE " + attack + ": " + str(eps)) + ece1 * 100) / 2

                    metric_dic["Total Accuracy " + attack + ": " + str(eps)] = (metric_dic.get(
                        "Total Accuracy " + attack + ": " + str(eps)) + acc2 * 100) / 2
                    metric_dic["Total ECE " + attack + ": " + str(eps)] = (metric_dic.get(
                        "Total ECE " + attack + ": " + str(eps)) + ece2 * 100) / 2
                else:
                    metric_dic["Accumulated Accuracy " + attack + ": " + str(eps)] = acc1 * 100
                    metric_dic["Accumulated ECE " + attack + ": " + str(eps)] = ece1 * 100

                    metric_dic["Total Accuracy " + attack + ": " + str(eps)] = acc2 * 100
                    metric_dic["Total ECE " + attack + ": " + str(eps)] = ece2 * 100


def calculateFrobeniusGradient(model, dataloader, device, voting_strategy):
    torch.cuda.empty_cache()

    if voting_strategy == "hard":
        model = EnsembleVoting(voting_strategy="soft", models=model.models).to(device)

    model.eval()
    fro_sum = torchmetrics.aggregation.SumMetric().to(device)

    for imgs, labels in dataloader:
        torch.cuda.empty_cache()
        # Move data to device
        imgs, labels = imgs.to(device), labels.to(device)
        imgs.requires_grad_()

        # Forward pass to compute the gradients
        pred = model(imgs)
        ce_loss = F.cross_entropy(pred, labels)

        # Compute gradient of cross-entropy loss w.r.t. input images
        grad_outputs = torch.ones_like(ce_loss)
        gradient = torch.autograd.grad(outputs=ce_loss, inputs=imgs, grad_outputs=grad_outputs, create_graph=True)[0]

        # Compute squared sum of gradients for each example in the batch
        gradient_squared_sum = torch.sum(gradient ** 2, dim=(1, 2, 3))  # Assuming imgs is 4D tensor

        frobenius_norm = torch.mean(gradient_squared_sum)

        fro_sum(torch.sqrt(frobenius_norm))

    return torch.sqrt(fro_sum.compute()) / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    torch.cuda.empty_cache()

    ece = torchmetrics.classification.MulticlassCalibrationError(num_classes=15).to(device)
    acc = torchmetrics.classification.MulticlassAccuracy(num_classes=15).to(device)
    eval_loss = 0

    model.eval()
    with torch.inference_mode():
        for imgs, labels in dataloader:
            # Move data to device
            imgs, labels = imgs.to(device), labels.to(device)

            # Predict
            pred = model(imgs)
            ece(pred, labels)
            acc(pred, labels)

            # Compute the loss and save for visualization
            loss = F.cross_entropy(pred, labels)
            eval_loss += loss.item()
    return acc.compute().item(), eval_loss / len(dataloader.dataset), ece.compute().item()


def calculate_weights(model, dataloader, eps, device, voting_strategy):
    model.weights = []
    for estimator in model.models:
        if voting_strategy == "weighted_robust":
            model.weights.append(evaluate_adv_attack(estimator, dataloader, "FGSM", eps, device)[0])
        else:
            weight, _, _ = evaluate(estimator, dataloader, device)
            model.weights.append(weight)


if __name__ == "__main__":
    inittrain()
