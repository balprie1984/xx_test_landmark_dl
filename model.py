import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2= nn.MaxPool2d(2, 1)
        self.fc1 = nn.Linear(16 * 12* 12 , 1024)
        self.fc2 = nn.Linear(1024, 224)
        self.fc3 = nn.Linear(224, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.pool(F.relu(self.conv1(x))) # o/p torch.Size([50, 16, 112, 112])
        x = self.pool(F.relu(self.conv2(x))) # o/p torch.Size([50, 32, 56, 56])
        x = self.pool(F.relu(self.conv3(x))) # o/p torch.Size([50, 64, 28, 28])
        x = self.pool(F.relu(self.conv4(x))) # o/p torch.Size([50, 32, 14, 14])
        x = self.pool2(F.relu(self.conv5(x))) # 0/p torch.Size([50, 64, 13, 13])
        x = self.pool2(F.relu(self.conv6(x))) # 0/p torch.Size([50, 16, 12, 12])
        x = x.view(-1, 16 * 12* 12 )
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
