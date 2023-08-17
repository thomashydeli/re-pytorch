from tqdm.notebook import tqdm
import torch
class vggTrainingInfer:

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        device='cpu',
    ):
        """Initialization function for the training functions

        Args:
          model: a PyTorch neural network model
          loss_fn: chosen loss function
          optimizer: optimzer for training the NN
          device: on cpu or gpu
        """
        self.model=model
        self.loss_fn=loss_fn
        self.optimizer=optimizer
        self.device=device


    def _step(
        self,
        dataLoader,
        infer=False,
    ):
        if infer:
            self.model.eval()
        else:
            self.model.train()

        currentLoss, currentMetric=0, 0
        for batch, (X,y) in enumerate(dataLoader):
            X,y=X.to(self.device), y.to(self.device)
            y_pred=self.model(X)
            loss=self.loss_fn(y_pred, y)
            currentLoss += loss.item()

            if not infer:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            y_pred_output=torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            currentMetric+=(y_pred_output==y).sum().item()/len(y_pred)

        currentLoss=currentLoss/len(dataLoader)
        currentMetric=currentMetric/len(dataLoader)
        return currentLoss, currentMetric


    def fullStep(
        self,
        train_dataloader,
        test_dataloader,
        epochs,
    ):
        """Where actual training and evaluation being performed

        Args:
          train_dataloader: DataLoader with training dataset
          test_dataloader: DataLoader with testing dataset
          epochs: number of epochs for training
        """
        results={
            'train_loss':[],
            'train_metric':[],
            'eval_loss':[],
            'eval_metric':[],
        }
        for epoch in tqdm(range(epochs), total=epochs):
            train_loss, train_acc=self._step(
                train_dataloader,
            )

            test_loss, test_acc=self._step(
                test_dataloader,
                infer=True,
            )
            print(f"Epoch: {epoch} | train loss: {train_loss} | train_acc: {train_acc}")
            print(f"test loss: {test_loss} | test_acc: {test_acc}")
            results['train_loss'].append(train_loss)
            results['train_metric'].append(train_acc)
            results['eval_loss'].append(test_loss)
            results['eval_metric'].append(test_acc)
        return results


def build_engine(
  model,
  loss_fn,
  optimizer,
  train_dataloader,
  test_dataloader,
  epochs,
  device='cpu'
):
  """Actual function which model training class will be called
  and Where actual training and evaluation being performed

  Args:
    model: a PyTorch neural network model
    loss_fn: chosen loss function
    optimizer: optimzer for training the NN
    train_dataloader: DataLoader with training dataset
    test_dataloader: DataLoader with testing dataset
    epochs: number of epochs for training
    device: on cpu or gpu

  Returns:
    A trained pytorch model with evaluated results containing accuracy and losses

  Example usage:
    trained_model, loss_summary = build_engine(
      model,
      torch.nn.CrossEntropyLoss(),
      torch.optim.Adam(model.parameters(),lr=5e-4),
      trainDataLoader,
      testDataLoader,
      epochs=2,
    )
  """
  trainer=vggTrainingInfer(
    model,
    loss_fn,
    optimizer,
    device,
  )
  results=trainer.fullStep(
    train_dataloader,
    test_dataloader,
    epochs
  )
  return (trainer.model, results)