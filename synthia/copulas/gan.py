from typing import Literal, Union
from .copula import Copula
import torch
from torch.nn import Module
import numpy as np

class GANCopula(Copula, Module):
    """
    Learns Copula from data using Generative Adversarial Networks.
    """
    def __init__(
            self,
            generator_deep_layers: int = 2,
            discriminator_deep_layers: int = 2,
            device: Literal['cpu', 'cuda', 'auto'] = 'auto',
            generator_optimizer: Literal['adam', 'sgd'] = 'adam',
            discriminator_optimizer: Literal['adam', 'sgd'] = 'adam',
        )->None:
        """
        Args:
            generator_deep_layers (int): Number of layers for the generator.
            discriminator_deep_layers (int): Number of layers for the discriminator.
            device (Literal['cpu', 'cuda', 'auto']): Device to use for training. Use 'auto' to automatically select the device.
            generator_optimizer (Literal['adam', 'sgd']): Optimizer to use for training the generator.
            discriminator_optimizer (Literal['adam', 'sgd']): Optimizer to use for training the discriminator.
        
        Returns:
            None
        """
        Module.__init__(self)
        Copula.__init__(self)

        self.device = None
        match device:
            case 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            case 'cpu':
                self.device = 'cpu'
            case 'cuda':
                self.device = 'cuda'
            case _:
                raise ValueError(f"Invalid device {device}. Use 'cpu', 'cuda' or 'auto'.")

        self.n_gen = generator_deep_layers
        self.n_dis = discriminator_deep_layers
        self.g_opt = generator_optimizer
        self.d_opt = discriminator_optimizer
        self.loss = torch.nn.BCELoss()
    
    def get_optimizer(self, optimizer_type: Literal['adam', 'sgd'], lr) -> torch.optim.Optimizer:
        """
        Get the optimizer for the generator or discriminator.

        Args:
            optimizer (Literal['adam', 'sgd']): The optimizer to get.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        match optimizer_type:
            case 'adam':
                return torch.optim.Adam(self.generator.parameters(), lr=lr)
            case 'sgd':
                return torch.optim.SGD(self.generator.parameters(), lr=lr)
            case _:
                raise ValueError(f"Invalid optimizer {optimizer_type}. Use 'adam' or 'sgd'.")

    
    def fit(
            self,
            X: Union[torch.Tensor, np.array],
            epochs: int = 1,
            lr: float = 0.001,
            batch_size: int = 32,
        ) -> None:
        """
        Fits the copula to data.
        
        Args:
            X (Union[torch.Tensor, np.array]): Input data in the shape (n_samples, n_features).
            epochs (int): Number of epochs to train the GAN.
            lr (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
        
        Returns:
            None
        """
        with self.device:
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            elif isinstance(X, torch.Tensor):
                X = X.float()
            else:
                raise TypeError(f"Invalid type {type(X)} for X. Use torch.Tensor or np.array.")

            #Create Layers
            self.generator = torch.nn.Sequential(
                torch.nn.Linear(1, X.shape[1]),
                torch.nn.ReLU(),
                *[layer for _ in range(self.n_gen) for layer in [torch.nn.Linear(X.shape[1], X.shape[1]), torch.nn.ReLU()]],
                torch.nn.Linear(X.shape[1], X.shape[1])
            )

            self.discriminator = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], X.shape[1]),
                torch.nn.ReLU(),
                *[layer for _ in range(self.n_dis) for layer in [torch.nn.Linear(X.shape[1], X.shape[1]), torch.nn.ReLU()]],
                torch.nn.Linear(X.shape[1], 1)
            )

            self.generator_optimizer = self.get_optimizer(self.g_opt, lr)
            self.discriminator_optimizer = self.get_optimizer(self.d_opt, lr)

            #Train GAN
            for epoch in range(epochs):
                for i in range(0, X.shape[0], batch_size):
                    #Train Discriminator
                    self.discriminator.zero_grad()
                    real = X[i:i+batch_size]
                    fake = self.generator(torch.rand(batch_size, 1))
                    loss = self.loss(self.discriminator(real), torch.ones(batch_size, 1)) +\
                          self.loss(self.discriminator(fake), torch.zeros(batch_size, 1))
                    loss.backward()
                    self.generator_optimizer.step()

                    #Train Generator
                    self.generator.zero_grad()
                    fake = self.generator(torch.rand(batch_size, 1))
                    loss = self.loss(self.discriminator(fake), torch.ones(batch_size, 1))
                    loss.backward()
                    self.discriminator_optimizer.step()
        
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generates samples from the copula.
        
        Args:
            n_samples (int): Number of samples to generate.
        
        Returns:
            np.ndarray: Samples from the copula.
        """
        with self.device:
            return self.generator(torch.rand(n_samples, 1)).detach().numpy()