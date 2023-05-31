import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import torch.optim as optim



# Basic fully connected neural network with MSE loss
class NN4vi(pl.LightningModule):
    """
     Creates a fully connected neural network
    :param input_dim: int, dimension of input data
    :param hidden_widths: list of size of hidden widths of network
    :param output_dim: dimension of output
    :param activation: activation function for hidden layers (defaults to ReLU)
    :return: A network
     """


    def __init__(self,
                 input_dim: int,
                 hidden_widths: list,
                 output_dim: int,
                 activation = nn.ReLU,
                 weight_decay=0,
                 lr=1e-3):
        super().__init__()
        structure = [input_dim] + list(hidden_widths) + [output_dim]
        layers = []
        for j in range(len(structure) - 1):
            act = activation if j < len(structure) - 2 else nn.Identity
            layers += [nn.Linear(structure[j], structure[j + 1]), act()]

        self.net = nn.Sequential(
            *layers
        )
        self.weight_decay=weight_decay
        self.lr=lr

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch
        y_hat = self.net(x)
        loss = nn.MSELoss()(y_hat, y)
        # Logging to TensorBoard by default
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = nn.MSELoss()(self.net(x), y)
        self.log('val_loss', loss)
        return loss

# Basic fully connected neural network with cross-entropy loss
class BinClass(pl.LightningModule):
    """
     Creates a fully connected 2-layer neural network with binary cross-entropy loss and ReLU activation
    :param p: int, dimension of input data
    :param hidden_width: int, size of hidden width of network

    :return: A network
     """
    def __init__(self, p=4, hidden_width=50):
        super().__init__()

        self.layer_1 = nn.Linear(p, hidden_width)
        self.layer_2 = nn.Linear(hidden_width, 1)


    def forward(self, x):
        # Pass the tensor through the layers
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)

        # Softmax the values to get a probability
        x = F.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Pass through the forward function of the network
        probs = self(x)
        loss = nn.BCELoss()(probs, y)
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = nn.BCELoss()(probs, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = nn.BCELoss()(probs, y)
        y_hat = torch.round(probs)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Very very simple network for MNIST classification
class UltraLiteMNIST(pl.LightningModule):
    """
     Creates a fully connected 2-layer neural network for images with binary cross-entropy loss and ReLU activation
    :param hidden_width: int, size of hidden width of network
    :param weight_decay: float, l2 regularization parameter

    :return: A network
     """
    def __init__(self, hidden_width=4, weight_decay=0):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, hidden_width)
        self.layer_3 = nn.Linear(hidden_width, 1)
        self.weight_decay = weight_decay

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch.tensor(x.view(batch_size, -1), dtype=torch.float32)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        logits = self(x)
        loss = nn.BCELoss()(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        probs = self(x)
        loss = nn.BCELoss()(probs, y)
        y_hat = torch.round(probs)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)

class LiteMNIST10(pl.LightningModule):
    """
     Creates a fully connected 2-layer neural network for images with binary cross-entropy loss and ReLU activation
    :param hidden_width: int, size of hidden width of network
    :param weight_decay: float, l2 regularization parameter

    :return: A network
     """
    def __init__(self, hidden_size=64, learning_rate=2e-4, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.dims = (1, 28, 28)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        channels, width, height = self.dims
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )
        self.accuracy = Accuracy()

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch.tensor(x.view(batch_size, -1), dtype=torch.float32)
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        #y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        #y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class LiteMNIST10_other1(pl.LightningModule):
    """
     Creates a fully connected 2-layer neural network for images with binary cross-entropy loss and ReLU activation
    :param hidden_width: int, size of hidden width of network
    :param weight_decay: float, l2 regularization parameter

    :return: A network
     """
    def __init__(self, hidden_size=64, learning_rate=2e-4):
        super().__init__()
        # Define PyTorch model
        self.num_classes = 10
        self.dims = (1, 28, 28)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        channels, width, height = self.dims
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )
        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



class MNIST10(pl.LightningModule):
    """
     Creates a fully connected 2-layer neural network for images with binary cross-entropy loss and ReLU activation
    :param hidden_width: int, size of hidden width of network
    :param weight_decay: float, l2 regularization parameter

    :return: A network
     """
    def __init__(self, hidden_size=64, learning_rate=2e-4, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.dims = (1, 28, 28)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        channels, width, height = self.dims
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 10)
        )
        self.accuracy = Accuracy()

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch.tensor(x.view(batch_size, -1), dtype=torch.float32)
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        #y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        #y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class MNIST10_sage(pl.LightningModule):
    """
     Creates a fully connected 2-layer neural network for images with binary cross-entropy loss and ReLU activation
    :param hidden_width: int, size of hidden width of network
    :param weight_decay: float, l2 regularization parameter

    :return: A network
     """
    def __init__(self, learning_rate=1e-3, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.dims = (1, 28, 28)
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        channels, width, height = self.dims
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 10)
        )
        self.accuracy = Accuracy()

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch.tensor(x.view(batch_size, -1), dtype=torch.float32)
        x = self.model(x)
        #return F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        #y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        #y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class MNIST10_cnn(pl.LightningModule):

    def __init__(self, in_channels=1, out_channels=10):
        super(MNIST10_cnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 8, 3, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(8, 16, 3, 1)
        # self.bn2 = nn.BatchNorm2d(16)

        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1352, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.conv1(x).clone()
        x = self.bn1(x).clone()
        x = self.relu(x).clone()
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        x = self.maxpool(x).clone()
        x = self.dropout1(x).clone()

        x = torch.flatten(x, 1).clone()
        x = self.fc1(x).clone()
        x = self.relu(x).clone()
        x = self.dropout2(x).clone()
        logits = self.fc2(x)
        return logits

    # define the loss function
    def criterion(self, logits, targets):
        return F.cross_entropy(logits, targets)

    # process inside the training loop
    def training_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)

        # inbuilt tensorboard for logs
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, nesterov=False, weight_decay=5e-4)
        return optimizer



class LiteMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x.clone()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        logits = self(x)
        loss =  nn.CrossEntropyLoss()(logits, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        probs = self(x)
        loss =  nn.CrossEntropyLoss()(probs, y)
        y_hat = torch.round(probs)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)