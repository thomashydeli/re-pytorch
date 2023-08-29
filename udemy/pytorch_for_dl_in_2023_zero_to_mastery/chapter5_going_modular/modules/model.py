"""
Contains functionality for creating a mock-up version of VGG
for image classification.
"""
import torch
from torch import nn

class VGGMock(nn.Module):
  def __init__(self, num_classes=3):
    """Initialization function for the VGGMock model

    Args:
      num_classes: an integer representing how many of the classes final output should be
    """
    super(VGGMock, self).__init__()

    self.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    self.avgpool = nn.AdaptiveAvgPool2d((8,8))

    self.classifier = nn.Sequential(
        nn.Linear(256 * 8 * 8, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(256, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(64, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x,1)
    x = self.classifier(x)
    return x


def model_build(num_classes):
  """Actual function for creating the VGG Mockup model

  Args:
    num_classes: an integer representing how many of the classes final output should be

  Returns:
    A pytorch model with the structure of mocked up VGG for classifying images into
    num_classes categories

  Example usage:
    model = model_build(
      num_classes=3
    )
  """

  model=VGGMock(num_classes)
  return model