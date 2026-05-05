from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from cnn import BaseCNN
import numpy as np
import torch
from copy import deepcopy
import os
from pathlib import Path
from tqdm import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning

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
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        metric: str,
        architecture_args: dict,
        shuffle: bool = True,
        patience=3, min_delta=0.005,
        task: str = "detection",
        num_classes: int = None,
    ):

        
        architecture = architecture_args.copy()
        if num_classes is not None:
            architecture["num_classes"] = num_classes
        self.cnn = BaseCNN(input_shape=input_shape, **architecture)
        architecture["input_shape"] = input_shape
        self._architecture = architecture


        # Get Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #to save the model
        self.results_path=results_path
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.loss_name = loss_function_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_epochs = num_epochs
        self.metric = metric
        self.task = task
        self.num_classes = num_classes

        #self._set_optimizer_and_loss() #now in the train loop

        #earlystopping
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

        #create the saving folder if doesn't exit
        if not os.path.exists(self.results_path) : 
            os.makedirs(self.results_path, exist_ok=True)

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

    def _set_optimizer_and_loss(self, class_weights=None):
        """Set the optimizer and loss function"""
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.cnn.parameters(), lr=self.learning_rate
            )
        else:
            raise NotImplementedError("Only Adam optimizer is supported at the moment")

        if self.loss_name == "cross_entropy":
            if class_weights is not None:
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
                self.criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            else:
                self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                "Only cross entropy loss is supported at the moment"
            )
        
    def _set_scheduler(self, patience=15):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=patience
        )

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)

    def get_minimum_input_shape(self):
        return self.cnn.calculate_min_input_size()

    def _create_dataloaders(
        self, X: np.ndarray, Y: np.ndarray, val_ratio: float = 0.2, seed: int = 42, num_workers: int = 0
    ):
        """Split X/Y into train and val DataLoaders (used when no separate val set is provided)."""
        X_tensor = torch.from_numpy(X).float()
        Y_tensor = torch.from_numpy(Y).long() if self.task == "classification" else torch.from_numpy(Y).float()

        if X_tensor.ndim == 3:
            X_tensor = X_tensor.unsqueeze(1)

        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(X_tensor), generator=generator)
        X_tensor = X_tensor[perm]
        Y_tensor = Y_tensor[perm]

        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        n_val = int(val_ratio * len(dataset))
        n_train = len(dataset) - n_val
        train_ds = torch.utils.data.Subset(dataset, range(0, n_train))
        val_ds = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

        loader_kwargs = dict(batch_size=self.batch_size)
        if num_workers > 0:
            loader_kwargs.update(num_workers=num_workers, pin_memory=True, persistent_workers=True)

        train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)
        return train_loader, val_loader

    def _create_dataloader(
        self, X: np.array, Y: np.array, num_workers : int,
    ) -> torch.utils.data.DataLoader:
        """Create a dataloader from the given data

        Parameters
        ----------
        X : np.array
            Input data of shape (n_samples,height,width) or (n_samples,channels,height,width). Will add channel dimension if needed.
        Y : np.array
            Target data of shape (n_samples,).

        Returns
        -------
        loader : torch.utils.data.DataLoader
            Dataloader with the given data and batch size specified in the constructor.

        """
        X_tensor = torch.from_numpy(X).float()
        Y_tensor = torch.from_numpy(Y).long() if self.task == "classification" else torch.from_numpy(Y).float()

        # Reshape X_tensor
        if len(X_tensor.shape) == 3:
            X_tensor = X_tensor.unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        
        if num_workers>0 :
            loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle,
            num_workers=num_workers, pin_memory=True,persistent_workers=True)
        else : 
            loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        
        return loader
    
    def _early_stop(self, val_loss, patience=10, min_delta=0.001):
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
        self.cnn.to(self.device)
        running_loss = 0.0

        num_batches = len(dataloader)
        num_samples = len(dataloader.dataset)

        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(self.device
                ), batch_targets.to(self.device)
            # Reset gradients
            self.optimizer.zero_grad()
            # Forward pass
            batch_preds = self.cnn.forward(batch_inputs)
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
        self.cnn.to(self.device)
        num_batches = len(dataloader)
        num_samples = len(dataloader.dataset)
    
        with torch.no_grad():
            running_loss = 0.0

            for batch_inputs, batch_targets in dataloader:
                batch_inputs, batch_targets = batch_inputs.to(
                    self.device
                ), batch_targets.to(self.device)

                # Forward pass
                batch_preds = self.cnn.forward(batch_inputs)
                # Compute loss
                loss = self.criterion(batch_preds, batch_targets)
                # Calculate Loss
                running_loss += loss.item() * batch_inputs.size(0)
      
            epoch_loss = running_loss / num_samples

        return epoch_loss

    def train(self, X_train, Y_train, X_val=None, Y_val=None, save=True, model_name="baseline", early_stopping=True, num_workers=0, patience=10, min_delta=0.001, class_weights=None, val_ratio=0.2, seed=42, scheduler_patience=15):
        
        # Set optimizer and loss (with optional weights)
        self._set_optimizer_and_loss(class_weights=class_weights)


        # Set scheduler (after optimizer exists)
        self._set_scheduler(patience=scheduler_patience)
        
        # Create Dataloaders
        if X_val is None or Y_val is None:
            train_loader, val_loader = self._create_dataloaders(
                X_train, Y_train, val_ratio=val_ratio, seed=seed, num_workers=num_workers
            )
        else:
            train_loader = self._create_dataloader(X_train, Y_train, num_workers=num_workers)
            val_loader = self._create_dataloader(X=X_val, Y=Y_val, num_workers=num_workers)
        val_losses=[]
        train_losses = []
        min_val_loss = torch.inf


        epoch_bar = tqdm(range(self.n_epochs), desc="Training", unit="epoch")

        for epoch in epoch_bar:
            epoch_train_loss=self._train_one_epoch(train_loader)
            train_losses.append(epoch_train_loss)
            
            epoch_val_loss = self._val_one_epoch(val_loader)
            val_losses.append(epoch_val_loss)

            # Save best model BEFORE scheduler step
            if save and epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                self.save_model(self.results_path, model_name)
            
            # NEW: update LR based on validation loss
            self.scheduler.step(epoch_val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update progress bar with current losses
            epoch_bar.set_postfix({
                'lr':         f'{current_lr:.2e}',
                'train_loss': f'{epoch_train_loss:.4f}',
                'val_loss':   f'{epoch_val_loss:.4f}',
                'best_val':   f'{min_val_loss:.4f}' if min_val_loss != torch.inf else 'N/A'
            })
            
            if early_stopping and self._early_stop(epoch_val_loss, patience=patience, min_delta=min_delta):
                tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        
        
        return train_losses, val_losses

    def evaluate(self, X_val, Y_val, metric=None, threshold=None, num_workers=0, print_report=False):
        if metric is None:
            metric = self.metric
        loader = self._create_dataloader(X=X_val, Y=Y_val,num_workers=num_workers)
        self.cnn.eval()

        with torch.no_grad():
            total_loss = 0
            targets = []
            predictions = []
            for batch_inputs, batch_targets in loader:
                batch_inputs, batch_targets = batch_inputs.to(
                    self.device
                ), batch_targets.to(self.device)
                # print("targets: ", batch_targets)
                batch_preds = self.cnn.forward(batch_inputs)
                total_loss += self.criterion(batch_preds, batch_targets).item()
                
                prediction = batch_preds.argmax(dim=1).cpu()
                if self.task == "detection":
                    if threshold is not None:
                        prediction = (batch_preds > threshold).float().cpu()
                    target = batch_targets.argmax(dim=1).cpu()
                else:
                    target = batch_targets.cpu()

                predictions.extend(prediction.detach().numpy())
                targets.extend(target.detach().numpy())

            f1_avg = "macro" if self.task == "classification" else "binary"
            f1 = f1_score(targets, predictions, average=f1_avg)
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
        self._model_state_dict = deepcopy(self.get_model_dict())
        torch.save(self._model_state_dict, save_path)
        #print(f"CNN model state dict saved to {save_path}!")

    def __call__(self, x):
        return self.cnn(x)

    def __str__(self):
        return str(self.cnn)

    def __repr__(self):
        return str(self.cnn)
