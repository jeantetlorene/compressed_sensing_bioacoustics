from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from cnn import BaseCNN
import numpy as np
import torch
from copy import deepcopy
import os
from pathlib import Path
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Suppress only UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)




class Model:
    """Model class."""

    def __init__(
        self,
        results_path,
        input_shape,
        optimizer_name: str,
        loss_function_name: str,
        num_classes: int,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        metric: str,
        architecture_args: dict,
        shuffle: bool = True,
        patience=3, min_delta=0.005
    ):

        
        architecture = architecture_args
        self.cnn = BaseCNN(input_shape=input_shape, **architecture)
        architecture = architecture.copy()
        architecture["input_shape"] = input_shape
        self._architecture = architecture

    
        # Get Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #to save the model 
        self.results_path=results_path
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.loss_name = loss_function_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_epochs = num_epochs
        self.metric = metric

        self._set_optimizer_and_loss()

        #earlystopping
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    @staticmethod
    def load_cnn(cnn_dict, device):
        """
        Load the model from a saved state dictionary of the CNN.

        Parameters
        ----------
        cnn_dict_path : str
            Path to the saved cnn model dictionary.

        Returns
        -------
        Model
            The loaded model.
        """
        # Check if its a path or a dictionary
        if type(cnn_dict) == dict:
            dictionary = cnn_dict
        else:
            if os.path.exists(cnn_dict):
                dictionary = torch.load(cnn_dict, map_location=device)
            else:
                raise FileNotFoundError(f"Model file {cnn_dict} not found")
        cnn = BaseCNN(**dictionary["architecture"])
        cnn.load_state_dict(dictionary["state_dict"])
        return cnn
    
    
    @staticmethod
    def load(self, model_dict):
        self.cnn = BaseCNN(**model_dict["architecture"])
        self.cnn.load_state_dict(model_dict["state_dict"])

  


    def get_model_dict(self):
        return {"state_dict": self.cnn.state_dict(), "architecture": self._architecture}

    def _set_optimizer_and_loss(self):
        """Set the optimizer and loss function"""
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.cnn.parameters(), lr=self.learning_rate
            )
        else:
            raise NotImplementedError("Only Adam optimizer is supported at the moment")

        if self.loss_name == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                "Only cross entropy loss is supported at the moment"
            )

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)

    def get_minimum_input_shape(self):
        return self.cnn.calculate_min_input_size()
   
    def _create_dataloaders(self,
        X: np.ndarray,
        Y: np.ndarray,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader]:

        # Convert to tensors
        X_tensor = torch.from_numpy(X).float()
        Y_tensor = torch.from_numpy(Y).long()

        # Add channel dimension if needed
        if X_tensor.ndim == 3:
            X_tensor = X_tensor.unsqueeze(1)

        # ---- EXPLICIT SHUFFLE BEFORE SPLIT ----
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(X_tensor), generator=generator)

        X_tensor = X_tensor[perm]
        Y_tensor = Y_tensor[perm]

        dataset = TensorDataset(X_tensor, Y_tensor)

        # ---- train / val split ----
        n_total = len(dataset)
        n_val = int(val_ratio * n_total)
        n_train = n_total - n_val

        train_dataset = torch.utils.data.Subset(dataset, range(0, n_train))
        val_dataset = torch.utils.data.Subset(dataset, range(n_train, n_total))

        # ---- DataLoaders ----
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,   # shuffle each epoch
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return train_loader, val_loader
    
    def _early_stop(self, val_loss, patience, min_delta):
        self.early_stop_counter = getattr(self, 'early_stop_counter', 0)
        self.best_val_loss = getattr(self, 'best_val_loss', torch.inf)
        
        if val_loss < (self.best_val_loss - min_delta):  # must improve by at least min_delta
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= patience:
                return True
            return False
    
    def _train_one_epoch(self, dataloader):
        self.cnn.train()
        running_loss = 0.0

        num_batches = len(dataloader)
        num_samples = len(dataloader.dataset)

        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(self.device
                ), batch_targets.to(self.device)
            # Reset gradients
            self.optimizer.zero_grad()
            # Forward pass
            batch_preds = self.cnn(batch_inputs)

            # Compute loss
            loss = self.criterion(batch_preds, batch_targets)
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            # Calculate Loss
            running_loss += loss.item() * batch_inputs.size(0)
        
        epoch_loss = running_loss / num_samples
        return epoch_loss
    
    def _val_one_epoch(self, dataloader):
        self.cnn.eval()
        num_batches = len(dataloader)
        num_samples = len(dataloader.dataset)
    
        with torch.no_grad():
            running_loss = 0.0

            for batch_inputs, batch_targets in dataloader:
                batch_inputs, batch_targets = batch_inputs.to(
                    self.device
                ), batch_targets.to(self.device)

                # Clear the gradients
                #self.optimizer.zero_grad()
                # Forward pass
                batch_preds = self.cnn(batch_inputs)
                # Compute loss
                loss = self.criterion(batch_preds, batch_targets)
                # Calculate Loss
                running_loss += loss.item() * batch_inputs.size(0)
      
            epoch_loss = running_loss / num_samples

        return epoch_loss

    def train(self, X, Y, save=True, model_name="baseline", early_stopping=True, resume_from=None, start_epoch=0, patience=10, min_delta=0.001 ):
        
        self.cnn.to(self.device)
        
        if resume_from and os.path.exists(resume_from):
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.cnn.load_state_dict(checkpoint["state_dict"])
            print(f"Resumed from {resume_from}")
        
        
        # Create Dataloaders
        train_loader, val_loader = self._create_dataloaders(X, Y)
        val_losses=[]
        train_losses = []
        min_val_loss = torch.inf

        for epoch in tqdm(range(self.n_epochs)):
            epoch_train_loss=self._train_one_epoch(train_loader)
            train_losses.append(epoch_train_loss)
            
            epoch_val_loss = self._val_one_epoch(val_loader)
            val_losses.append(epoch_val_loss)

            torch.cuda.empty_cache()  # clear cache each epoch

            # Save every 5 epochs regardless
            if save and epoch % 5 == 0:
                self.save_model(self.results_path, f"{model_name}_epoch{epoch}")
            
            # Save best
            if save and min_val_loss > epoch_val_loss:
                min_val_loss = epoch_val_loss
                self.save_model(self.results_path, model_name)
            

            if early_stopping and self._early_stop(epoch_val_loss, patience=patience, min_delta=min_delta):
                            tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
                            break
   
        return train_losses, val_losses

    def evaluate(self, X, Y, metric=None, threshold=None, print_report=False):
        if metric is None:
            metric = self.metric
        _, loader = self._create_dataloaders(X, Y)
        self.cnn.eval()

        targets = []
        predictions = []

        with torch.no_grad():

            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                # Forward pass (raw logits)
                logits = self.cnn(batch_inputs)

                # Multiclass prediction
                preds = logits.argmax(dim=1)

                predictions.extend(preds.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())

            f1 = f1_score(targets, predictions, average="macro")
            report = classification_report(targets, predictions)
            confusion = confusion_matrix(targets, predictions)
            if print_report:
                print(report)
                print(confusion)
            # print("F1: ", f1)
            # print("true positives: ", np.sum(np.logical_and(targets, predictions)))
            # print("true negatives: ", np.sum(np.logical_and(np.logical_not(targets), np.logical_not(predictions))))
            # print("false positives: ", np.sum(np.logical_and(np.logical_not(targets), predictions)))
            # print("false negatives: ", np.sum(np.logical_and(targets, np.logical_not(predictions))))
            accuracy = accuracy_score(targets, predictions)

        if metric == "f1":
            return (f1, "F1")
        elif metric == "accuracy":
            return (accuracy, "Accuracy")

    def save_model(self, path, model_name):
        save_path = os.path.join(Path(path, model_name + "_cnn_state.pth"))
        model_dict = self.get_model_dict()
        model_dict["state_dict"] = {k: v.detach().cpu() for k, v in model_dict["state_dict"].items()}

        torch.save(model_dict, save_path)
        #print(f"CNN model state dict saved to {save_path}!")
        

    def __call__(self, x):
        return self.cnn(x)

    def __str__(self):
        return str(self.cnn)

    def __repr__(self):
        return str(self.cnn)
