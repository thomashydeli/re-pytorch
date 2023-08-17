"""
Contains functionality to download and preprocess data to support subsequent ML model
training and testing
"""
import os
import zipfile
import requests
from pathlib import Path
from logging import Logger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional, List, Any, Tuple


# function to perform data preservation and downloading
def get_data(
    data_path: str,
    data_url: str,
    unzip: Optional[bool] = False,
    logger: Optional[Logger] = None,
):
    """Actual function to download data into the data_path given
    the url to download data.

    Args:
        data_path: Path to preserve the data
        data_url: URL to perform downloading of the data
        unzip: Optional, whether the data should be unpacked
        logger: Optional, logger to preserve key module information

    Returns:
        A tuple of (train_dir, test_dir) with downloaded data preserved to
        the corresponding data_path directory

    Example usage:
        train_dir, test_dir = get_data(
            data_path='data/pizza_steak_sushi',
            data_url='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip',
            unzip=True,
        )
    """

    pth=Path(data_path) # creating pth as a Posix path
    data_pth=pth.parent.absolute()

    # creating folder for data preservation if not exists
    if pth.is_dir():
        if logger: logger.info(f'{pth} exists')
    else:
        if logger: logger.info(f'Did not find {pth} directory, creating...')
        pth.mkdir(parents=True, exist_ok=True)

    # downloading data
    fname='downloaded_file.zip' if unzip else 'downloaded_file'
    with open(data_pth / fname,'wb') as f:
        request = requests.get(data_url)
        if logger: logger.info('image data being downloaded ...')
        f.write(request.content)

    # unzipping packagelized data if needed
    if unzip:
        with zipfile.ZipFile(data_pth / fname,'r') as zip_ref:
            if logger: logger.info('Unzipping packagelized image data ...')
            zip_ref.extractall(pth)

    # removing the zip file:
    os.remove(data_pth / fname)

    # returning two paths
    return (pth / 'train', pth / 'test')


# function to create dataloaders for training and testing datasets
def create_dataloaders(
  train_dir: str,
  test_dir: str,
  transformations: List[Any],
  img_size: Tuple[int],
  batch_size: int,
  num_workers: int=os.cpu_count(),
  logger: Optional[Logger]=None,
):
    """Actual function for turning data folders into create_dataloader
    from training and testing directories

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transformations: List of PyTorch image transformations with the type of torchvision.transforms
                    for variation in training datasets
        img_size: Size of the image which desired to be for uniformity (post-transformations)
        batch_size: Number of images in a batch for training and testing
        num_workers: an integer for number of workers per DataLoader
        logger: Optional, logger for information purpose

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names, class_dict)
        class_name is a list of the target classes
        class_dict is the indexer corresponding to the target classes

    Example usage:
        train_dataloader, test_dataloader, class_names, class_dict = create_dataloaders(
            train_dir=path/to/train_dir,
            test_dir=path/to/test_dir,
            img_size=(64,64),
            transformations=[transforms.RandomHorizontalFlip(0.2)],
            batch_size=16,
            num_workers=4,
        )
    """
    # setting up the transformations
    training_transformations=[transforms.Resize(size=(img_size, img_size))] + transformations + [transforms.ToTensor()]
    testing_transformations=[transforms.Resize(size=(img_size, img_size)), transforms.ToTensor()]


    trainDataset=datasets.ImageFolder(
        root=train_dir,
        transform=transforms.Compose(training_transformations),
    ) # creating training dataset
    if logger: logger.info('training dataset set up')

    trainDataLoader=DataLoader(
        trainDataset,
        batch_size=batch_size,
        shuffle=True,
    ) # creating training dataloader
    if logger: logger.info('training dataloader set up')

    # getting the name and indexer of classes:
    class_name=trainDataset.classes
    class_dict=trainDataset.class_to_idx
    if logger: logger.debug(f'classes are: {class_name}')

    # setting up the testing dataset
    testDataset=datasets.ImageFolder(
        root=test_dir,
        transform=transforms.Compose(testing_transformations),
    ) # creating testing dataset
    if logger: logger.info('testing dataset set up')

    # creating the testing dataloader
    testDataLoader=DataLoader(
        testDataset,
        batch_size=batch_size,
        shuffle=False,
    ) # creating testing dataloader
    if logger: logger.info('testing dataloader set up')

    return (
        trainDataLoader,
        testDataLoader,
        class_name,
        class_dict
    )