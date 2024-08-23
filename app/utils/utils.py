
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torchattacks import FGSM, PGD

from Data.dataset import CustomDataset
from Models.EnsembleVoting import EnsembleVoting
from Training.regularizers import Chi_Square_Distance, GradientIndependence, SampleDiverstiyAdversarial, \
    NegativeCorrelationNC, \
    SampleDiversity, ADP, NegativeCorrelation, Gradient_Regularizer, Jacobian_Regularizer


def configure_regularizer(model, regularizer, device, alpha, beta, eps, voting_strategy):
    if voting_strategy == "hard":
        model = EnsembleVoting(voting_strategy="soft", models=model.models).to(device)

    if regularizer == "NC":
        regularizer = NegativeCorrelation(model=model)
    elif regularizer == "ADP":
        regularizer = ADP(alpha=alpha, beta=beta, device=device, model=model)
    elif regularizer == "SD":
        regularizer = SampleDiversity(model=model, device=device)
    elif regularizer == "NC_NCORRECT":
        regularizer = NegativeCorrelationNC(device=device, model=model)
    elif regularizer == "SD_ADV":
        regularizer = SampleDiverstiyAdversarial(model=model, eps=eps, device=device)
    elif regularizer == "GI":
        regularizer = GradientIndependence(model=model)
    elif regularizer == "DJR":
        regularizer = Jacobian_Regularizer(model=model)
    elif regularizer == "DGR":
        regularizer = Gradient_Regularizer(model=model)
    elif regularizer == "IGR":
        regularizer = []
        for estimator in model.models:
            regularizer.append(Gradient_Regularizer(model=estimator))
    elif regularizer == "IJR":
        regularizer = []
        for estimator in model.models:
            regularizer.append(Jacobian_Regularizer(model=estimator))
    elif regularizer == "CHI":
        regularizer = Chi_Square_Distance(model=model)
    elif regularizer == "":
        regularizer = None
    else:
        raise TypeError

    return regularizer


def getModel(model="resnet", voting_strategy="soft", device="cuda:0"):
    implemented_models = {"resnet", "densnet", "shufflenet", "ensemble"}
    if model not in implemented_models:
        msg = (
            "Regularizer {} is not implemented, "
            "please choose from {}."
        )
        raise ValueError(
            msg.format(model, implemented_models)
        )

    if model == "densnet":
        model = torchvision.models.densenet121()
        model.classifier = nn.Sequential(
            nn.Linear(1024, 15),
            nn.Softmax(dim=1)
        )

    elif model == "resnet":
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(
            nn.Linear(512, 15),
            nn.Softmax(dim=1)
        )

    elif model == "shufflenet":
        model = torchvision.models.shufflenet_v2_x0_5()
        model.fc = nn.Sequential(
            nn.Linear(1024, 15),
            nn.Softmax(dim=1)
        )

    else:
        resnet = torchvision.models.resnet18()
        resnet.fc = nn.Sequential(
            nn.Linear(512, 15),
            nn.Softmax(dim=1)
        )

        densnet = torchvision.models.densenet121()
        densnet.classifier = nn.Sequential(
            nn.Linear(1024, 15),
            nn.Softmax(dim=1)
        )

        shufflenet = torchvision.models.shufflenet_v2_x0_5()
        shufflenet.fc = nn.Sequential(
            nn.Linear(1024, 15),
            nn.Softmax(dim=1)
        )

        model = EnsembleVoting(models=[resnet.to(device), densnet.to(device), shufflenet.to(device)],
                               voting_strategy=voting_strategy)

    return model.to(device)


def getDataloader(batchsize=128, numworkers=0, ids=[]):
    if len(ids) == 0:
        raise TypeError

    sampler = torch.utils.data.SubsetRandomSampler(ids)
    dataset = CustomDataset()

    return torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batchsize,
        num_workers=numworkers,
        persistent_workers=True)


def getOptimizer(modelInstance, weight_decay, lr, model):
    optimizers = []
    if model == "ensemble":
        for estimator in modelInstance.models:
            optimizer = torch.optim.Adam(estimator.parameters(), lr=lr, weight_decay=weight_decay)
            optimizers.append(optimizer)
    else:
        optimizers.append(torch.optim.Adam(modelInstance.parameters(), lr=lr, weight_decay=weight_decay))

    return optimizers


def getDataLoaders(train_val_ids, test_ids, batchsize=128, numworkers=0):
    # Define data loaders for training and testing data in this fold
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.25)
    dataloader_train = getDataloader(batchsize=batchsize, numworkers=numworkers, ids=train_ids)
    dataloader_val = getDataloader(batchsize=batchsize, numworkers=numworkers, ids=val_ids)
    dataloader_test = getDataloader(batchsize=batchsize, numworkers=numworkers, ids=test_ids)

    return dataloader_train, dataloader_val, dataloader_test


def getNormStdDataset(dataset=None):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=200,
        num_workers=7,
        shuffle=False
    )

    mean = 0.0
    for batch, (images, _) in enumerate(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * 50 * 50))

    print(mean)
    print(std)

    return mean, std


def writeResultsEpoch(writer, n_iter, result, tag):
    print(tag + ": " + str(result))
    writer.add_scalar(tag, result, n_iter)


def writeTestResults(writer, hparam_dict, metric_dict):
    for x in metric_dict:
        print(x + ": " + str(metric_dict[x]))
    writer.add_hparams(hparam_dict, metric_dict)


def visualizeFeatures(ensemble_model, device, dataloader, regularizer, model, voting_strategy, initial_dims=30, no_dims=2):
    # Init Soft-Model if Voting-Strategy = Hard for backpropagting Gradients
    if voting_strategy == "hard":
        ensemble_model = EnsembleVoting(voting_strategy="soft", models=ensemble_model.models).to(device)

    ensemble_model.to(device)

    # Extract features from each model separately
    extracted_features_resnet = []
    extracted_features_densnet = []
    extracted_features_shufflenet = []
    all_labels = []

    # FGSM attack parameters
    epsilon = 0.05
    attack = FGSM(ensemble_model, eps=epsilon)
    attack.set_device(device)

    # Forward pass through the ensemble model using the data loader
    for batch in dataloader:
        torch.cuda.empty_cache()
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial images
        adversarial_images = attack(images, labels)

        # Define hook functions inside the loop
        def hook_fn_resnet(module, input, output):
            extracted_features_resnet.append(input[0].detach().cpu())

        def hook_fn_densnet(module, input, output):
            extracted_features_densnet.append(input[0].detach().cpu())

        def hook_fn_shufflenet(module, input, output):
            extracted_features_shufflenet.append(input[0].detach().cpu())

        # Register hooks after generating adversarial images
        hook_handles = [
            ensemble_model.models[0].layer4.register_forward_hook(hook_fn_resnet),
            ensemble_model.models[1].features.register_forward_hook(hook_fn_densnet),
            ensemble_model.models[2].conv5.register_forward_hook(hook_fn_shufflenet)
        ]

        # Forward pass through the ensemble model using adversarial images
        ensemble_model(adversarial_images)

        # Unregister hooks after the forward pass
        for handle in hook_handles:
            handle.remove()

        all_labels.append(labels.cpu().detach().numpy())

    # Concatenate all_labels three times
    all_labels_concatenated = np.concatenate(all_labels + all_labels + all_labels)

    # Concatenate features from all models
    features_resnet = torch.cat(extracted_features_resnet, dim=0)
    features_densnet = torch.cat(extracted_features_densnet, dim=0)
    features_shufflenet = torch.cat(extracted_features_shufflenet, dim=0)

    # Flatten the tensors before concatenating
    features_resnet = torch.flatten(features_resnet, start_dim=1)
    features_densnet = torch.flatten(features_densnet, start_dim=1)
    features_shufflenet = torch.flatten(features_shufflenet, start_dim=1)

    min_dimension = min(features_resnet.shape[1], features_densnet.shape[1], features_shufflenet.shape[1])
    pca_resnet = PCA(n_components=min_dimension)
    pca_densnet = PCA(n_components=min_dimension)
    pca_shufflenet = PCA(n_components=min_dimension)
    pca_features_resnet = pca_resnet.fit_transform(features_resnet.cpu().numpy())
    pca_features_densnet = pca_densnet.fit_transform(features_densnet.cpu().numpy())
    pca_features_shufflenet = pca_shufflenet.fit_transform(features_shufflenet.cpu().numpy())

    # Perform PCA on the concatenated features
    pca = PCA(n_components=initial_dims)
    pca_features_all = np.concatenate([pca_features_resnet, pca_features_densnet, pca_features_shufflenet], axis=0)
    pca_features = pca.fit_transform(pca_features_all)

    # Perform symmetric t-SNE on the PCA-transformed features
    tsne = TSNE(n_components=no_dims, perplexity=30)
    features_2d_all = tsne.fit_transform(pca_features)

    # Visualize the features as a scatter plot

    plt.figure(figsize=(10, 8))

    # Plot features from ResNet in blue
    plt.scatter(features_2d_all[:len(features_resnet), 0], features_2d_all[:len(features_resnet), 1], label='ResNet', alpha=0.2)

    # Plot features from DenseNet in red
    plt.scatter(features_2d_all[len(features_resnet):len(features_resnet)+len(features_densnet), 0], features_2d_all[len(features_resnet):len(features_resnet)+len(features_densnet), 1], label='DenseNet', alpha=0.2)

    # Plot features from ShuffleNet in green
    plt.scatter(features_2d_all[-len(features_shufflenet):, 0], features_2d_all[-len(features_shufflenet):, 1], label='ShuffleNet', alpha=0.2)

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Extracted Features from Ensemble Model')
    plt.legend()
    plt.savefig('Features_' + model + "_" + regularizer + "_" + voting_strategy + '.png')


def visualize_feature_distribution(ensemble_model, device, dataloader, regularizer, model, voting_strategy):
    """
    Visualize feature distribution using a violin plot.

    Parameters:
        model_type (str): Type of base model (e.g., "resnet", "densnet", "shufflenet").
        device (str): Device to run the model on (e.g., "cuda:0", "cpu").
        dataloader (DataLoader): DataLoader containing the dataset.
        regularizer (str): Type of regularizer used.
        model (str): Type of ensemble model (e.g., "soft", "hard").
        voting_strategy (str): Voting strategy used by the ensemble model.

    Returns:
        None
    """
    # Init Soft-Model if Voting-Strategy = Hard for backpropagting Gradients
    if voting_strategy == "hard":
        ensemble_model = EnsembleVoting(voting_strategy="soft", models=ensemble_model.models).to(device)

    ensemble_model.to(device)

    # Extract features from each model separately
    extracted_features_resnet = []
    extracted_features_densnet = []
    extracted_features_shufflenet = []
    all_labels = []

    # FGSM attack parameters
    epsilon = 0.05
    attack = FGSM(ensemble_model, eps=epsilon)
    attack.set_device(device)

    # Forward pass through the ensemble model using the data loader
    for batch in dataloader:
        torch.cuda.empty_cache()
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial images
        adversarial_images = attack(images, labels)

        # Define hook functions inside the loop
        def hook_fn_resnet(module, input, output):
            extracted_features_resnet.append(input[0].detach().cpu())

        def hook_fn_densnet(module, input, output):
            extracted_features_densnet.append(input[0].detach().cpu())

        def hook_fn_shufflenet(module, input, output):
            extracted_features_shufflenet.append(input[0].detach().cpu())

        # Register hooks after generating adversarial images
        hook_handles = [
            ensemble_model.models[0].layer4.register_forward_hook(hook_fn_resnet),
            ensemble_model.models[1].features.register_forward_hook(hook_fn_densnet),
            ensemble_model.models[2].conv5.register_forward_hook(hook_fn_shufflenet)
        ]

        # Forward pass through the ensemble model using adversarial images
        ensemble_model(adversarial_images)

        # Unregister hooks after the forward pass
        for handle in hook_handles:
            handle.remove()

        all_labels.append(labels.cpu().detach().numpy())

    # Concatenate all_labels three times
    all_labels_concatenated = np.concatenate(all_labels + all_labels + all_labels)

    # Concatenate features from all models
    features_resnet = torch.cat(extracted_features_resnet, dim=0)
    features_densnet = torch.cat(extracted_features_densnet, dim=0)
    features_shufflenet = torch.cat(extracted_features_shufflenet, dim=0)

    # Flatten the tensors before concatenating
    features_resnet = torch.flatten(features_resnet, start_dim=1)
    features_densnet = torch.flatten(features_densnet, start_dim=1)
    features_shufflenet = torch.flatten(features_shufflenet, start_dim=1)

    min_dimension = min(features_resnet.shape[1], features_densnet.shape[1], features_shufflenet.shape[1])
    pca_resnet = PCA(n_components=min_dimension)
    pca_densnet = PCA(n_components=min_dimension)
    pca_shufflenet = PCA(n_components=min_dimension)
    pca_features_resnet = pca_resnet.fit_transform(features_resnet.cpu().numpy())
    pca_features_densnet = pca_densnet.fit_transform(features_densnet.cpu().numpy())
    pca_features_shufflenet = pca_shufflenet.fit_transform(features_shufflenet.cpu().numpy())
    pca_features_all = np.concatenate([pca_features_resnet, pca_features_densnet, pca_features_shufflenet], axis=0)

    # Create a DataFrame for easy visualization
    pca_features_df = pd.DataFrame(pca_features_all, columns=[f"PC{i}" for i in range(pca_features_all.shape[1])])
    pca_features_df["Class"] = all_labels_concatenated

    # Visualize feature distribution by class
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=pca_features_df, x="Class", y="PC0", hue="Class", split=True, inner="quart")
    plt.title("Feature Distribution of Ensemble Model by Class")
    plt.xlabel("Class")
    plt.ylabel("Principal Component 0")
    plt.xticks(rotation=45)
    plt.savefig('Features_Distribution_' + model + "_" + regularizer + "_" + voting_strategy + '.png')


def visualize_feature_distribution_hard(ensemble_model, device, dataloader, regularizer, model, voting_strategy):
    """
    Visualize feature distribution using a violin plot.

    Parameters:
        ensemble_model: The ensemble model.
        device (str): Device to run the model on (e.g., "cuda:0", "cpu").
        dataloader (DataLoader): DataLoader containing the dataset.
        regularizer (str): Type of regularizer used.
        model (str): Type of ensemble model (e.g., "soft", "hard").
        voting_strategy (str): Voting strategy used by the ensemble model.

    Returns:
        None
    """
    # Init Soft-Model if Voting-Strategy = Hard for backpropagting Gradients
    if voting_strategy == "hard":
        ensemble_model = EnsembleVoting(voting_strategy="soft", models=ensemble_model.models).to(device)

    ensemble_model.to(device)

    # Extract features from each model separately
    extracted_features_resnet = []
    extracted_features_densnet = []
    extracted_features_shufflenet = []
    all_labels = []

    # FGSM attack parameters
    epsilon = 0.05
    attack = FGSM(ensemble_model, eps=epsilon)
    attack.set_device(device)

    # Forward pass through the ensemble model using the data loader
    for batch in dataloader:
        torch.cuda.empty_cache()
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial images
        adversarial_images = attack(images, labels)

        # Define hook functions inside the loop
        def hook_fn_resnet(module, input, output):
            extracted_features_resnet.append(input[0].detach().cpu())

        def hook_fn_densnet(module, input, output):
            extracted_features_densnet.append(input[0].detach().cpu())

        def hook_fn_shufflenet(module, input, output):
            extracted_features_shufflenet.append(input[0].detach().cpu())

        # Register hooks after generating adversarial images
        hook_handles = [
            ensemble_model.models[0].layer4.register_forward_hook(hook_fn_resnet),
            ensemble_model.models[1].features.register_forward_hook(hook_fn_densnet),
            ensemble_model.models[2].conv5.register_forward_hook(hook_fn_shufflenet)
        ]

        # Forward pass through the ensemble model using adversarial images
        ensemble_model(adversarial_images)

        # Unregister hooks after the forward pass
        for handle in hook_handles:
            handle.remove()

        all_labels.append(labels.cpu().detach().numpy())

    # Concatenate all_labels three times
    all_labels_concatenated = np.concatenate(all_labels)

    # Concatenate features from all models
    features_resnet = torch.cat(extracted_features_resnet, dim=0)
    features_densnet = torch.cat(extracted_features_densnet, dim=0)
    features_shufflenet = torch.cat(extracted_features_shufflenet, dim=0)

    # Flatten the tensors before concatenating
    features_resnet = torch.flatten(features_resnet, start_dim=1)
    features_densnet = torch.flatten(features_densnet, start_dim=1)
    features_shufflenet = torch.flatten(features_shufflenet, start_dim=1)

    # Perform PCA on the concatenated features
    min_dimension = min(features_resnet.shape[1], features_densnet.shape[1], features_shufflenet.shape[1])
    pca_resnet = PCA(n_components=min_dimension)
    pca_densnet = PCA(n_components=min_dimension)
    pca_shufflenet = PCA(n_components=min_dimension)
    pca_features_resnet = pca_resnet.fit_transform(features_resnet.cpu().numpy())
    pca_features_densnet = pca_densnet.fit_transform(features_densnet.cpu().numpy())
    pca_features_shufflenet = pca_shufflenet.fit_transform(features_shufflenet.cpu().numpy())

    # Create a DataFrame for easy visualization
    pca_features_df_resnet = pd.DataFrame(pca_features_resnet,
                                          columns=[f"PC{i}" for i in range(pca_features_resnet.shape[1])])
    pca_features_df_resnet["Class"] = all_labels_concatenated

    pca_features_df_densnet = pd.DataFrame(pca_features_densnet,
                                           columns=[f"PC{i}" for i in range(pca_features_densnet.shape[1])])
    pca_features_df_densnet["Class"] = all_labels_concatenated

    pca_features_df_shufflenet = pd.DataFrame(pca_features_shufflenet,
                                              columns=[f"PC{i}" for i in range(pca_features_shufflenet.shape[1])])
    pca_features_df_shufflenet["Class"] = all_labels_concatenated

    # Visualize feature distribution by class for each model separately
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Plot violin plots for each model separately with custom colors and transparency
    sns.violinplot(data=pca_features_df_resnet, x="Class", y="PC0", color='blue', label='ResNet', split=True,
                   inner="quart", alpha=0.5)
    sns.violinplot(data=pca_features_df_densnet, x="Class", y="PC0", color='red', label='DenseNet', split=True,
                   inner="quart", alpha=0.5)
    sns.violinplot(data=pca_features_df_shufflenet, x="Class", y="PC0", color='green', label='ShuffleNet', split=True,
                   inner="quart", alpha=0.5)

    plt.title("Feature Distribution of Ensemble Model by Class")
    plt.xlabel("Class")
    plt.ylabel("Principal Component 0")
    plt.xticks(rotation=45)

    # Add custom legend
    custom_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='ResNet'),
                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,
                                label='DenseNet'),
                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10,
                                label='ShuffleNet')]
    plt.legend(handles=custom_legend)

    plt.savefig('Features_Distribution_' + model + "_" + regularizer + "_" + voting_strategy + '.png')


def visualize_gradient(ensemble_model, dataloader, device, regularizer, model, voting_strategy):

    # Init Soft-Model if Voting-Strategy = Hard for backpropagting Gradients
    if voting_strategy == "hard":
        ensemble_model = EnsembleVoting(voting_strategy="soft", models=ensemble_model.models).to(device)

    # Set model to evaluation mode
    ensemble_model.eval()

    # Placeholder for gradients
    all_gradients = []

    # Iterate through the dataloader
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True  # Ensure gradients are computed for images

        # Forward pass
        outputs = ensemble_model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Backward pass
        ensemble_model.zero_grad()  # Zero gradients before backward pass
        loss.backward()

        # Get gradients
        gradients = images.grad.detach().cpu().numpy()

        # Append gradients to the list
        all_gradients.append(gradients)

    # Concatenate gradients from all batches
    all_gradients = np.concatenate(all_gradients, axis=0)

    # Calculate magnitude of gradients
    magnitude = np.sqrt(np.sum(all_gradients ** 2, axis=1))  # Compute magnitude across channels

    # Reshape the magnitude array to (num_samples, height, width)
    magnitude = magnitude.reshape((-1, images.shape[-2], images.shape[-1]))

    # Visualize the magnitude as a heatmap
    plt.figure()
    plt.imshow(magnitude.mean(axis=0), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap of Gradient Magnitude')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig('Gradient_' + model + "_" + regularizer + "_" + voting_strategy + '.png')














