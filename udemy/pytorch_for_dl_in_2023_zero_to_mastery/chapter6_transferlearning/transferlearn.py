import torch
import numpy as np
import torchvision
from torch import nn
from PIL import Image
from torchinfo import summary
import matplotlib.pyplot as plt
from torchvision import transforms
from modularized.engine import vggTrainingInfer, build_engine
from modularized.data_processes import get_data, create_dataloaders

def fine_tune(
    model,
    weights,
    loss_fn,
    optimizer,
    last_layer=2048,
    batch_size=16,
    num_workers=4,
    epochs=20,
    learning_rate=1e-4,
    data_path='data/pizza_steak_sushi',
    data_url='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip',
    device='cpu',
    unzip=True,
):

    train_dir, test_dir=get_data(
        data_path=data_path,
        data_url=data_url,
        unzip=unzip,
    )
    auto_transforms=weights.transforms()

    train_dataloader, test_dataloader, class_names, class_dict = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        img_size=(
            auto_transforms.crop_size[0], auto_transforms.crop_size[0]
        ),
        transformations=auto_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model=model.to(device)

    for param in model.parameters():
        param.requires_grad=False

    # Update the classifier head of our model to suit the problem
    model.fc=nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(
            in_features=last_layer,
            out_features=len(class_names)
        )
    ).to(device)

    # setting up loss function and optimizer
    optimizer=optimizer(
        model.parameters(),
        lr=learning_rate,
    )

    # create engine and train
    print('performing training of the model')
    model, results=build_engine(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=epochs,
        device=device
    )

    return model, test_dataloader


def pred_and_plot_image(
    model,
    image_path,
    class_names,
    image_size,
    weights,
    device='cpu',
):
    img=Image.open(image_path)
    
    model.to(device)
    model.eval()
    transform=weights.transforms()
    with torch.inference_mode():
        transformed_image=transform(img).unsqueeze(dim=0)
        target_image_pred=model(transformed_image.to(device))
    
    target_image_pred_probs=torch.softmax(target_image_pred, dim=1)
    target_image_pred_label=torch.argmax(target_image_pred_probs, dim=1).item()
    target_image_pred_probs=target_image_pred_probs.numpy()

    plt.figure()
    plt.imshow(img)
    plt.title(f'Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs[0][target_image_pred_label]*100:.3f}%')