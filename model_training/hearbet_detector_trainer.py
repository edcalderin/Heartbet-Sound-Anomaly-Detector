from pathlib import Path
from typing import Callable, Dict, List, Tuple
from torch import nn
import torch
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import logging

class HearbetDetectorTrainer:

    def __init__(self, model: nn.Module, device: torch.device, model_dir: str) -> None:

        if not isinstance(model, nn.Module):
            raise ValueError('The provided model is not an instance of nn.Module')

        self.__device = device
        self.__precision_fn: Callable = Accuracy(task = 'binary').to(device)
        self.__model = model.to(device)
        self.__model_dir = model_dir

    def __inner_loop(self, specs, labels, loss_function: torch.nn.Module, is_train: bool) -> Tuple:
        specs, labels = specs.to(self.__device), labels.float().to(self.__device)
        outputs = self.__model(specs).squeeze()
        loss = loss_function(outputs, labels.squeeze())
        loss_value = loss.item()

        if is_train:
            loss.backward()

        return loss_value, self.__precision_fn(outputs, labels).item()

    def __train_step(self,
                     optimizer: torch.optim.Optimizer,
                     train_loader: torch.utils.data.DataLoader,
                     loss_function: torch.nn.Module,
                     scheduler) -> Tuple[float, float]:

        train_loss, train_accuracy = 0, 0

        self.__model.train()
        for specs, labels in train_loader:
            optimizer.zero_grad()
            loss_value, accuracy_value = self.__inner_loop(specs, labels, loss_function, is_train = True)
            train_loss += loss_value
            train_accuracy += accuracy_value
            optimizer.step()

        scheduler.step()

        length_train_loader = len(train_loader)

        return train_loss/length_train_loader, train_accuracy/length_train_loader

    def __test_step(self, val_loader: torch.utils.data.DataLoader, loss_function: torch.nn.Module) -> Tuple[float, float]:
        val_accuracy, val_loss = 0, 0

        self.__model.eval()
        with torch.inference_mode():
            for specs, labels in val_loader:
                loss_value, accuracy_value = self.__inner_loop(specs, labels, loss_function, is_train = False)
                val_loss += loss_value
                val_accuracy += accuracy_value

        length_val_loader = len(val_loader)

        return val_loss/length_val_loader, val_accuracy/length_val_loader

    def fit(self,
            loss_function: nn.Module,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            batch_size: int,
            scheduler,
            training_set,
            validation_set) -> Dict[str, List]:

        logging.info('Training on', self.__device)

        train_loader = DataLoader(training_set, batch_size = batch_size, shuffle = True)
        val_loader = DataLoader(validation_set, batch_size = batch_size)

        report_dict: Dict =  {
            'loss': [],
            'precision': [],
            'val_loss': [],
            'val_precision': []
        }

        for epoch in range(epochs):
            train_loss, train_accuracy = self.__train_step(optimizer, train_loader, loss_function, scheduler)
            val_loss, val_accuracy = self.__test_step(val_loader, loss_function)

            model_name: str = f'epoch_{epoch}_acc={train_accuracy:.4f}_val_acc={val_accuracy:.4f}.pth'
            model_path = Path(self.__model_dir)
            model_path.mkdir(exist_ok = True)
            torch.save(self.__model.cpu().state_dict(), model_path.joinpath(model_name))

            self.__model.to(self.__device)

            report_dict['loss'].append(train_loss)
            report_dict['precision'].append(train_accuracy)
            report_dict['val_loss'].append(val_loss)
            report_dict['val_precision'].append(val_accuracy)

            logging.info('Epoch: {} | Loss: {:.4f} - Accuracy: {:.4f} - Val loss: {:.4f} - Val accuracy: {:.4f}'
                  .format(epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy))

        return report_dict
