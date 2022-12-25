import numpy as np
import pandas as pd

import torch
import torchvision

import matplotlib.pyplot as plt


# Pass this as a transform to dataset
class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()

        y_hat = model(x)
        
        loss = loss_fn(y_hat, y)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item()
    
    return train_step


def train(train_step_fn, train_loader, train_losses, train_counter, epoch, log_interval, train_batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        x_train =  torch.unsqueeze(x_train.to(device), dim=1)
        y_train =  torch.unsqueeze(y_train.to(device), dim=1)

        loss = train_step_fn(x_train, y_train)
        
        if batch_idx % log_interval == 0:
            train_losses.append(loss)
            train_counter.append((epoch-1) * len(train_loader.dataset) + batch_idx * len(x_train))
            
            print(f"Training epoch #{epoch} ({batch_idx * train_batch_size + len(x_train)}"\
                  f"/ {len(train_loader.dataset)}) ({(100. * batch_idx / len(train_loader)):.0f}%)"\
                  f" - Loss = {loss:.6f}")
    print("")
            

def test(model, loss_fn, test_loader, test_losses, epoch):
    test_loss = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = torch.unsqueeze(x_test.to(device), dim=1)
            y_test =  torch.unsqueeze(y_test.to(device), dim=1)

            model.eval()

            y_hat = model(x_test)
            
            test_loss += loss_fn(y_hat, y_test).item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if epoch == 0:
            print(f"Test - No training yet - Avg. loss = {test_loss:.6f}")
        else:
            print(f"Test epoch #{epoch} - Avg. loss = {test_loss:.6f}")
            
        print("---------------------------------------------\n\n")


class MNIST_AE(torch.utils.data.Dataset): # type: ignore    
    def __init__(self, training_data=True, classes=None, noise_mean=0, noise_std=0) -> None:
        mean = 0.1307
        std = 0.3081

        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.data = torchvision.datasets.MNIST(
                '/files/', 
                train=training_data,
                transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((mean,), (std,)),
                ]),
                download=True
        )

        if classes is not None:
            l = []
            for i in classes:
                l.append((self.data.targets == i).tolist())

            idx = [any(el) for el in zip(*l)]

            self.data.targets = self.data.targets[idx]
            self.data.data = self.data.data[idx]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        assert isinstance(idx, int)

        tr_x = torchvision.transforms.Compose([
                AddGaussianNoise(self.noise_mean, self.noise_std),
        ])

        x = tr_x(self.data[idx][0])[0]
        y = self.data[idx][0][0]

        return x, y
