import pickle
import torch
from pathlib import Path
from typing import List

def save_model(
  model: torch.nn.Module,
  classes: List[str],
  target_dir: str,
  model_name: str
):
  """Helper function for saving the model
  
  Args:
    model: trained PyTorch model
    classes: list of strings representing the classes of the images
    target_dir: dir for preserving the model
    model_name: name of the model

  Example usage:
    save_model(
        model,
        'model',
        'pizza_steak_sushi',
    )
  """

  target_dir_path=Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  # create model save pth
  assert model_name.endswith('.pth') or model_name.endswith('.pt')
  model_save_path = target_dir_path / model_name

  torch.save(model, model_save_path)

  with open(target_dir_path / 'classes.pickle', 'wb') as f:
    pickle.dump(classes,f)