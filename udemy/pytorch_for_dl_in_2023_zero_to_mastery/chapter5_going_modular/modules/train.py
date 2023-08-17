"""
Trains a PyTorch image classificaiton model
"""

import os
import torch
import argparse
from torchvision import transforms
from data_processes import get_data, create_dataloaders
from model import model_build
from utils import save_model
from engine import build_engine

parser = argparse.ArgumentParser(
    description="parsing variables for our classification model"
)
parser.add_argument('--data_pth', type=str, help="path for preserving data")
parser.add_argument('--data_url', type=str, help="url where the data is currently at")
parser.add_argument('--unzip', action='store_true', help="unpack downloaded data or not")
parser.add_argument('--image_size', type=int, help="size of the image")
parser.add_argument('--batch_size', type=int, help="batch size")
parser.add_argument('--learning_rate', type=float, help="model learning rate")
parser.add_argument('--epochs', type=int, help="number of epochs for training")
parser.add_argument('--model_dir', type=str, help="directory for saving the model")
parser.add_argument('--model_name', type=str, help="name of the model")

# parsing arguments
args=parser.parse_args()
DATA_PTH=args.data_pth
DATA_URL=args.data_url
UNZIP=args.unzip
IMAGE_SIZE=args.image_size
BATCH_SIZE=args.batch_size
LEARNING_RATE=args.learning_rate
EPOCHS=args.epochs
MODEL_DIR=args.model_dir
MODEL_NAME=args.model_name


# DATA_PTH='data/pizza_steak_sushi'
# DATA_URL='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip'
# UNZIP=True
# IMAGE_SIZE=64
# BATCH_SIZE=16
# LEARNING_RATE=0.001
# EPOCHS=5
# MODEL_DIR='model'
# MODEL_NAME='pizza_steak_sushi_model.pth'


# getting data and setting up training and testing directories
train_dir, test_dir = get_data(
    data_path=DATA_PTH,
    data_url=DATA_URL,
    unzip=UNZIP,
)


# preparing dataloaders
train_dataloader, test_dataloader, class_names, class_dict = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    img_size=IMAGE_SIZE,
    transformations=[transforms.RandomHorizontalFlip(0.2)],
    batch_size=BATCH_SIZE,
)
print('data processed successfully!')


# set up model:
model=model_build(len(class_names))
print('model set up successfully!')


# setting up loss function and optimizer
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(
  model.parameters(),
  lr=LEARNING_RATE
)


# create engine and train
print('performing training of the model')
model, results=build_engine(
  model=model,
  loss_fn=loss_fn,
  optimizer=optimizer,
  train_dataloader=train_dataloader,
  test_dataloader=test_dataloader,
  epochs=EPOCHS,
  device='cuda' if torch.cuda.is_available() else 'cpu'
)


# saving the model
save_model(
  model=model,
  classes=class_names,
  target_dir=MODEL_DIR,
  model_name=MODEL_NAME,
)
print('model has been saved')