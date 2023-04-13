import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

# Define the time series model using PyTorch Lightning
class TimeSeriesModel(pl.LightningModule):
    def __init__(self):
        super(TimeSeriesModel, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters())

# Generate a random time series dataset
time = np.arange(1000)
value = np.sin(time * 0.1) + np.random.randn(1000) * 0.1

# Split the dataset into training and validation sets
split_time = 700
train_time = time[:split_time]
train_value = value[:split_time]
valid_time = time[split_time:]
valid_value = value[split_time:]

# Create PyTorch DataLoader objects for the training and validation sets
train_dataset = TensorDataset(torch.from_numpy(train_time).float().unsqueeze(1), torch.from_numpy(train_value).float().unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataset = TensorDataset(torch.from_numpy(valid_time).float().unsqueeze(1), torch.from_numpy(valid_value).float().unsqueeze(1))
valid_loader = DataLoader(valid_dataset, batch_size=32)

# Train the model with different epochs to demonstrate underfitting and overfitting
model = TimeSeriesModel()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader, valid_loader)
trainer_overfit = pl.Trainer(max_epochs=50)
trainer_overfit.fit(model, train_loader, valid_loader)
trainer_underfit = pl.Trainer(max_epochs=1)
trainer_underfit.fit(model, train_loader, valid_loader)

# Plot the training and validation loss for each epoch
import matplotlib.pyplot as plt

def plot_loss(trainer):
    train_losses = [x['train_loss'] for x in trainer.logged_metrics]
    val_losses = [x['val_loss'] for x in trainer.logged_metrics]
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.legend()
    plt.show()

plot_loss(trainer)
plot_loss(trainer_overfit)
plot_loss(trainer_underfit)
