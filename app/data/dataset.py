import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class CustomDataset(Dataset):

    def __init__(self):
        self.transform = self.getTransformer()
        self.path = "C:/Users/tobia/PycharmProjects/2023_tobias_kaiser/Data/TRMSD"
        self.labels = pd.read_csv(self.path + "/labels.csv")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.path + "/imgs", self.labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        # Typecasting
        image = image.float()

        return image, label

    def getTransformer(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((50, 50)),
            transforms.ToTensor()
        ])
