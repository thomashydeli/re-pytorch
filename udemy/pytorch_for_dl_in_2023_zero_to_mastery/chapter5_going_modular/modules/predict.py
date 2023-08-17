"""Loads a Pytorch image classification model
and make a prediction"""

import pickle
import torch
import requests
import argparse
import torchvision
from pathlib import Path
from torchvision import transforms


parser = argparse.ArgumentParser(
    description="parsing variables for our classification model"
)
parser.add_argument('--img_file', type=str, help="name of the image to be infered")
parser.add_argument('--eval_dir', type=str, help="directory of the image to be infered")
parser.add_argument('--img_url', type=str, help="url of the image to be downloaded")
parser.add_argument('--img_size', type=int, help="size of the image")
parser.add_argument('--model_pth', type=str, help="directory for the saved model")
parser.add_argument('--model_name', type=str, help="name of the model")

# parsing arguments
args=parser.parse_args()
IMG_FILE=args.img_file
EVAL_DIR=args.eval_dir
IMG_URL=args.img_url
IMG_SIZE=args.img_size
MODEL_PTH=Path(args.model_pth)
MODEL_NAME=args.model_name

# IMG_FILE='04-pizza-dad.jpeg'
# EVAL_DIR='data/eval'
# IMG_URL='https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg'
# IMG_SIZE=64
# MODEL_PTH=Path('model')
# MODEL_NAME='pizza_steak_sushi.pth'

model=torch.load(MODEL_PTH / MODEL_NAME)
with open(MODEL_PTH / 'classes.pickle','rb') as f:
    class_names=pickle.load(f)
print('model loaded succesfully!')


# downloading image
eval_dir=Path(EVAL_DIR)
eval_dir.mkdir(parents=True, exist_ok=True)
custom_image_path=eval_dir / IMG_FILE
with open(custom_image_path,'wb') as f:
    request=requests.get(IMG_URL)
    print(f'downloading {custom_image_path}...')
    f.write(request.content)
print('image to infer has been downloaded successfully!')


# perform inference on the image downloaded
custom_image=torchvision.io.read_image(str(custom_image_path)).type(torch.float32)/255.
custom_image_transform=transforms.Compose(
    [transforms.Resize([IMG_SIZE, IMG_SIZE])]
)
custom_image_transformed=custom_image_transform(custom_image)
model.eval()
with torch.inference_mode():
  custom_image_pred=model(custom_image_transformed.unsqueeze(0))

predicted_class=class_names[custom_image_pred.argmax(dim=-1).item()]
print(f'class predicted as: {predicted_class}')