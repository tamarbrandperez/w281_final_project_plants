import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from typing import Tuple


class SimpleCNNClassifier:
    """
    A simple CNN classifier class using PyTorch.

    Attributes:
        train_dir (str): Directory path for training data.
        test_dir (str): Directory path for testing data.
        image_size (Tuple[int, int]): Size to which input images are resized.
        num_classes (int): Number of output classes for classification.
        batch_size (int): Number of samples per batch.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to perform computations (CPU or GPU).
    """

    def __init__(self,
                 train_dir: str,
                 test_dir: str,
                 image_size: Tuple[int, int],
                 num_classes: int,
                 batch_size: int,
                 learning_rate: float,
                 num_epochs: int) -> None:
        """
        Initializes the SimpleCNNClassifier with the given hyperparameters and directories.

        Args:
            train_dir (str): Path to the training data directory.
            test_dir (str): Path to the testing data directory.
            image_size (Tuple[int, int]): Target size for input images.
            num_classes (int): Number of classes in the dataset.
            batch_size (int): Batch size for training and testing.
            learning_rate (float): Learning rate for the optimizer.
            num_epochs (int): Number of training epochs.
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the CNN model
        self.model = self.SimpleCNN(num_classes).to(self.device)

        # Define the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        # Initialize data loaders
        self.train_loader, self.test_loader = self.prepare_data()

    class SimpleCNN(nn.Module):
        """
        A simple CNN model with two convolutional layers, one pooling layer, and two fully connected layers.
        """

        def __init__(self, num_classes: int) -> None:
            """
            Initializes the SimpleCNN model.

            Args:
                num_classes (int): Number of classes in the output layer.
            """
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Adjust dimensions based on image size after conv layers
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the network.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after forward pass.
            """
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32 * 8 * 8)  # Adjust dimensions based on input image size
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepares the data loaders for training and testing data.

        Returns:
            Tuple[DataLoader, DataLoader]: Train and test data loaders.
        """
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = datasets.ImageFolder(root=self.train_dir, transform=transform)
        test_data = datasets.ImageFolder(root=self.test_dir, transform=transform)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def train(self) -> None:
        """
        Trains the model using the training dataset for a specified number of epochs.
        """
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {running_loss / len(self.train_loader):.4f}")

    def evaluate(self) -> float:
        """
        Evaluates the model on the testing dataset and computes the accuracy.

        Returns:
            float: Accuracy of the model on the test data.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on test images: {accuracy:.2f}%')
        return accuracy

    def run(self) -> None:
        """
        Runs the complete training and evaluation process for the CNN model.
        """
        print("Starting training...")
        self.train()
        print("Training complete. Evaluating on test data...")
        accuracy = self.evaluate()
        print(f"Final accuracy: {accuracy:.2f}%")

# Example usage:
# cnn_classifier = SimpleCNNClassifier(
#     train_dir="path/to/train",
#     test_dir="path/to/test",
#     image_size=(32, 32),
#     num_classes=10,
#     batch_size=32,
#     learning_rate=0.001,
#     num_epochs=10
# )
# cnn_classifier.run()
