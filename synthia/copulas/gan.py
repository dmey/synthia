from typing import Literal, Union
from .copula import Copula
import torch
from torch.nn import Module
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class GANCopula(Copula, Module):
    """
    Learns Copula from data using Generative Adversarial Networks.
    """
    def __init__(
            self,
            generator_deep_layers: list[int] = [32, 32],
            discriminator_deep_layers: list[int] = [32, 32],
            device: Literal['cpu', 'cuda', 'auto'] = 'auto',
            optimizer: Literal['adam', 'sgd'] = 'adam',
        )->None:
        """
        Args:
            generator_deep_layers (list[int]): Number of deep layers for the generator. (Default: [32, 32])
            discriminator_deep_layers (list[int]): Number of deep layers for the discriminator. (Default: [32, 32])
            device (Literal['cpu', 'cuda', 'auto']): Device to use for training. Use 'auto' to automatically select the device.
            optimizer (Literal['adam', 'sgd']): Optimizer to use for training.
        
        Returns:
            None
        """
        Module.__init__(self)
        Copula.__init__(self)

        self.device = None
        match device:
            case 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            case 'cpu':
                self.device = torch.device('cpu')
            case 'cuda':
                self.device = torch.device('cuda')
            case _:
                raise ValueError(f"Invalid device {device}. Use 'cpu', 'cuda' or 'auto'.")

        self.n_gen = generator_deep_layers
        self.n_dis = discriminator_deep_layers
        self.opt = optimizer
        self.loss = torch.nn.BCEWithLogitsLoss()
    
    def get_optimizer(self, optimizer_type: Literal['adam', 'sgd'], lr: float) -> torch.optim.Optimizer:
        """
        Get the optimizer for the generator or discriminator.

        Args:
            optimizer (Literal['adam', 'sgd']): The optimizer to get.
            lr (float): The learning rate for the optimizer.

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
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fits the copula to data.
        
        Args:
            X (Union[torch.Tensor, np.array]): Input data in the shape (n_samples, n_features).
            epochs (int): Number of epochs to train the GAN.
            lr (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Generator and discriminator loss.
        """
        loss = None
        with self.device:
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            elif isinstance(X, torch.Tensor):
                X = X.float()
            else:
                raise TypeError(f"Invalid type {type(X)} for X. Use torch.Tensor or np.array.")
            X = X.to(self.device)
            self.n_features = X.shape[1]

            #Create Generator
            self.deep_gen_layers = [2] + self.n_gen + [self.n_features]
            self.deep_gen_layers = zip(self.deep_gen_layers[:-1], self.deep_gen_layers[1:])
            self.generator = torch.nn.Sequential(
                *[x for i, j in self.deep_gen_layers for x in [torch.nn.Linear(i, j), torch.nn.ReLU()]],
            )

            #Create Discriminator
            self.deep_dis_layers = [self.n_features] + self.n_dis + [1]
            self.deep_dis_layers = zip(self.deep_dis_layers[:-1], self.deep_dis_layers[1:])
            self.discriminator = torch.nn.Sequential(
                *[x for i,j in self.deep_dis_layers for x in [torch.nn.Linear(i, j), torch.nn.ReLU()]]
            )

            self.generator_optimizer = self.get_optimizer(self.opt, lr)
            self.discriminator_optimizer = self.get_optimizer(self.opt, lr)

            #Train GAN
            for epoch in range(epochs):
                for i in range(0, X.shape[0], batch_size):

                    real = X[i:i+batch_size]
                    #At the end the batch size might be smaller than the specified batch size
                    actual_batch_size = real.shape[0]
                    fake = self.generator(torch.rand(actual_batch_size, 2))

                    self.discriminator.zero_grad()
                    disc_real = self.discriminator(real)
                    disc_fake = self.discriminator(fake)

                    loss_real = self.loss(disc_real, torch.ones(actual_batch_size))
                    #Inserting 1 for y effectively calculates log(D(x)), as the second term on
                    #the loss vanishes as it is proportional to (1-y)
                    loss_real.backward()

                    loss_fake = self.loss(disc_fake, torch.zeros(actual_batch_size))
                    #In a similar fashion, inserting 0 for y effectively calculates log(1-D(G(z)))
                    loss_fake.backward()
                    loss_discriminator = loss_real + loss_fake

                    self.discriminator_optimizer.step()

                    self.generator.zero_grad()
                    new_disc_fake = self.discriminator(fake)
                    loss_generator = self.loss(new_disc_fake, torch.ones(actual_batch_size))
                    loss_generator.backward()
                    self.generator_optimizer.step()
                    

        
        return loss_discriminator, loss_generator
        
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generates samples from the copula.
        
        Args:
            n_samples (int): Number of samples to generate.
        
        Returns:
            np.ndarray: Samples from the copula.
        """
        with self.device:
            return self.generator(torch.rand(n_samples, 2)).detach().cpu().numpy()