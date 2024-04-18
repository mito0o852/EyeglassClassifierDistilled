import os
import torch
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

class ImageDatasetLoader:
    def __init__(self, directory, batch_size=2):
        """
        Initialize the ImageDatasetLoader with a specific directory.

        Args:
            directory (str): The directory where the images are stored.
            batch_size (int, optional): The batch size for the DataLoader. Defaults to 32.
        """
        self.directory = directory
        self.batch_size = batch_size

        # Define the transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the images to 224x224 pixels
            transforms.ToTensor()
        ])

    def load_images(self):
        """
        Load the images from the directory.

        Returns:
            torch.utils.data.DataLoader: The DataLoader for the images.
        """
        # Load the images from the directory
        dataset = datasets.ImageFolder(self.directory, transform=self.transform)

        # Create a data loader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader



class UnlabeledImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        super(UnlabeledImageDataset, self).__init__()
        self.directory = directory
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the images to 224x224 pixels
        ])
        self.image_paths = os.listdir(directory)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.image_paths[idx])
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image
