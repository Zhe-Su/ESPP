import os
import shutil
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm.notebook import trange

from data import (augment_shd, load_classwise_NMNIST, load_classwise_PMNIST,
                  load_SHD)
from logger import get_logger
from model import EchoSpike


class Trainer:
    """
    A trainer class for Spiking Neural Networks (SNNs) using the EchoSpike learning algorithm.
    """
    
    def __init__(self, cfg):
        """
        Initialize the Trainer.
        
        Args:
            cfg (DictConfig): Configuration dictionary containing all training parameters
        """
        # Set c_y based on online mode
        if hasattr(cfg, 'c_y_offline') and not cfg.online:
            cfg.c_y = cfg.c_y_offline
        
        # Store configuration
        self.cfg = cfg
        self.device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)
        self.batch_size = cfg.batch_size
        self.lr = cfg.lr
        self.online = cfg.online
        self.augment = cfg.augment
        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')

        # By default use the 'data' directory where the code is run from.
        # Overwritten in case of resuming from a previous run.
        self.data_dir = Path(to_absolute_path('data'))
        self.resume = cfg.common.resume
        
        # Setup logger
        self.logger = get_logger(name='trainer')
        
        self.logger.info(f"Initializing trainer with device: {self.device}")
        self.logger.info(f"Current working directory: {os.getcwd()}")
        
        # Setup directory structure
        self._setup_directories()

        # Create datasets
        self.train_loader, self.test_loader = self._create_datasets()
        
        # Create model
        self.SNN = EchoSpike(cfg.n_inputs, cfg.n_hidden, c_y=cfg.c_y, beta=cfg.beta,
                            device=cfg.device, recurrency_type=cfg.recurrency_type,
                            n_time_steps=cfg.n_time_bins, online=cfg.online, inp_thr=cfg.inp_thr)
        self.SNN = self.SNN.to(self.device)
        self.SNN.reset(0)  # after sending to cuda the espp_layers' mem is not on cuda, resetting fixes that
    
    def _create_datasets(self):
        """
        Create train and test datasets based on configuration.
        
        Returns:
            tuple: (train_loader, test_loader)
        """

        self.logger.info(f"Searching for cached dataset files in {self.data_dir}")
        if self.cfg.dataset == 'nmnist':
            train_loader, _, test_loader = load_classwise_NMNIST(
                self.cfg.n_time_bins, split_train=True, data_dir=self.data_dir, batch_size=self.cfg.batch_size
            )
        elif self.cfg.dataset == 'pmnist':
            train_loader, test_loader = load_classwise_PMNIST(
                self.cfg.n_time_bins, scale=self.cfg.poisson_scale, data_dir=self.data_dir, batch_size=self.cfg.batch_size
            )
        elif self.cfg.dataset == 'shd':
            train_loader, test_loader = load_SHD(data_dir=self.data_dir, batch_size=self.cfg.batch_size)
        else:
            raise ValueError(f"Unknown dataset: {self.cfg.dataset}")
        
        self.logger.info(f"Created datasets for {self.cfg.dataset} with batch size {self.cfg.batch_size}")
        return train_loader, test_loader

    def _setup_directories(self):
        """
        Create directory structure and copy files if not resuming.
        """

        if not self.resume:
            config_dir = Path('config')
            config_path = config_dir / 'config.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "config"), dst="./config", dirs_exist_ok=True)
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
        else:
            assert self.ckpt_dir.exists(), "Checkpoint directory does not exist for resuming."
            assert self.media_dir.exists(), "Media directory does not exist for resuming."
            original_hydra_path = Path(".hydra/hydra.yaml")
            assert original_hydra_path.exists(), "Missing hydra config to resume from."
            original_hydra_config = OmegaConf.load(original_hydra_path)
            self.data_dir = Path(original_hydra_config.hydra.runtime.cwd) / 'data'
     
    def save_checkpoint(self, epoch):
        """
        Save model checkpoint.

        Args:
            epoch (int): Epoch number for intermediate checkpoints.
        """
        checkpoint_path = self.ckpt_dir / f'{self.cfg.model_name}_epoch_{epoch}.pt'
        checkpoint = {'model_state_dict': self.SNN.state_dict(),
                      'epoch': epoch,}
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint at epoch {epoch}: {checkpoint_path}")
        
    def save_loss_history(self, loss_hist):
        """
        Save loss history.
        
        Args:
            loss_hist (torch.Tensor): Loss history tensor
            model_name (str): Name of the model
        """
        loss_path = self.ckpt_dir / f'{self.cfg.model_name}_loss_hist.pt'
        torch.save(loss_hist, loss_path)
        self.logger.info(f"Saved loss history: {loss_path}")

    def train(self, freeze=[]):
        """
        Trains a SNN.

        Args:
            freeze (list): List of layers to freeze during training.

        Returns:
            torch.Tensor: A tensor containing the loss history during training.
        """
        torch.set_grad_enabled(False)
        loss_hist = []
        accuracies = []
        print_interval = 100 * self.batch_size if 'mnist' in self.cfg.dataset else 40 * self.batch_size
        
        self.logger.info(f"Starting training for {self.cfg.epochs} epochs")
        self.logger.info(f"Batch size: {self.batch_size}, Learning rate: {self.lr}")
        self.logger.info(f"Online mode: {self.online}, Augmentation: {self.augment}")
        
        # training loop
        optimizer = torch.optim.SGD([{"params": par.fc.parameters(), 'lr': self.lr} for par in self.SNN.layers])
        optimizer.zero_grad()
        self.SNN.train()
        bf = 0
        target = [torch.randint(self.train_loader.num_classes, (1,)).item() for _ in range(self.batch_size)]
        spks = torch.zeros(len(self.SNN.layers) + 1, device=self.device)
        
        while True:

            # Train loop
            data, target = self.train_loader.next_item(target, contrastive=(bf == -1))

            data = data.float().to(self.device)
            if self.augment:
                data = augment_shd(data)

            target = target.to(self.device)
            sample_loss = torch.zeros(len(self.SNN.layers), device=self.device)

            i = 0
            for step in range(data.shape[0]):
                # iterate over time steps
                if self.online:
                    inp_activity = data[step].mean(axis=-1)
                else:
                    inp_activity = None
                spk, _, loss, grad = self.SNN(data[step], torch.tensor(bf, device=self.device), freeze, inp_activity=inp_activity)
                spks += torch.stack([data[step].mean(), *[sp.mean() for sp in spk]])    # to analyze nr of spks
                sample_loss += loss
                if self.online:
                    optimizer.step()
                    optimizer.zero_grad()
                i += 1
            loss_hist.append(sample_loss / data.shape[0]) 
            accuracies.append(self.SNN.reset(bf))

            if bf == -1 and not self.online:
                # update weights after one predictive and one contrastive batch, before weight update
                optimizer.step()
                optimizer.zero_grad()
            bf = 1 if bf != 1 else -1

            step = len(loss_hist) * self.batch_size
            epoch = step // len(self.train_loader)
            if step % print_interval < self.batch_size and len(loss_hist) > 1:
                # log loss and accuracy
                avg_loss = torch.stack(loss_hist[-print_interval//self.batch_size:]).mean(axis=0)
                avg_acc = torch.stack(accuracies).mean(axis=0)
                spikes = spks * self.batch_size / print_interval
                
                self.logger.info(f"Epoch {epoch}, Step {step}")
                self.logger.info(f"EchoSpike Loss: {avg_loss}")
                self.logger.info(f"Accuracy: {avg_acc}")
                self.logger.info(f"Spikes: {spikes}")
                
                accuracies = []
                spks = torch.zeros(len(self.SNN.layers) + 1, device=self.device)
            if epoch >= self.cfg.epochs:
                break
            if step % len(self.train_loader) < self.batch_size and epoch % 20 == 0:
                # save checkpoint
                current_epoch_loss = torch.stack(loss_hist[-20 * len(self.train_loader) // self.batch_size:]).mean().item()
                self.logger.info(f'Epoch {epoch} loss: {current_epoch_loss}')
                self.save_checkpoint(epoch)
        

        # Save final model and loss history
        loss_hist_tensor = torch.stack(loss_hist)
        self.save_loss_history(loss_hist_tensor)
        
        self.save_checkpoint(epoch)
        self.logger.info(f"Training completed. Total epochs: {self.cfg.epochs}")

        return loss_hist_tensor

    def test(self):
        """
        Tests a SNN.

        Returns:
            tuple: A tuple containing the following:
                - spk_history (list): A list of spike histories.
                - target_list (list): A list of target values.
                - losses (list): A list of loss values during testing.
        """
        torch.set_grad_enabled(False)
        self.SNN.eval()
        self.logger.info("Starting model evaluation")
        
        spk_history = []
        target_list = []
        losses = []

        bf = 0
        target = [torch.randint(self.test_loader.num_classes, (1,)).item() for _ in range(self.batch_size)]
        total_batches = int(len(self.test_loader) / self.batch_size)
        
        for batch_idx in trange(total_batches):
            data, target = self.test_loader.next_item(target, contrastive=(bf == -1))
            target_list.append(target)
            data = data.float().to(self.device)
            target = target.to(self.device)
            logit_list = []
            activation_list = []
            loss_sample = torch.zeros(len(self.SNN.layers), device=self.device)
            for step in range(data.shape[0]):
                out_spk, _, loss, _ = self.SNN(data[step], torch.tensor(bf, device=self.device))
                logit_list.append(out_spk[-1])
                activation_list.append(out_spk)
                loss_sample += loss

            losses.append(loss_sample)
            spk_history.append(activation_list[0])
            for i in range(1, len(activation_list)):
                for l in range(len(spk_history[-1])):
                    spk_history[-1][l] += activation_list[i][l]
            self.SNN.reset(bf)
            bf = 1 if bf != 1 else -1
            # if len(losses)*batch_size > len(testloader):
            #     break
        
        self.logger.info(f"Evaluation completed. Processed {len(losses)} batches")
        return spk_history, target_list, losses
