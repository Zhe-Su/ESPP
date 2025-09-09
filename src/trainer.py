import torch
from data import augment_shd, load_classwise_NMNIST, load_classwise_PMNIST, load_SHD
import numpy as np
import shutil
import hydra
from pathlib import Path
from tqdm.notebook import trange
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
        self.resume = cfg.common.resume
        
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
        if self.cfg.dataset == 'nmnist':
            train_loader, _, test_loader = load_classwise_NMNIST(
                self.cfg.n_time_bins, split_train=True, batch_size=self.cfg.batch_size
            )
        elif self.cfg.dataset == 'pmnist':
            train_loader, test_loader = load_classwise_PMNIST(
                self.cfg.n_time_bins, scale=self.cfg.poisson_scale, batch_size=self.cfg.batch_size
            )
        elif self.cfg.dataset == 'shd':
            train_loader, test_loader = load_SHD(batch_size=self.cfg.batch_size)
        else:
            raise ValueError(f"Unknown dataset: {self.cfg.dataset}")
            
        return train_loader, test_loader

    def _setup_directories(self):
        """
        Create directory structure and copy files if not resuming.
        """
        self.media_dir = Path('media')

        if not self.resume:
            import os
            print(os.getcwd())
            config_dir = Path('config')
            config_path = config_dir / 'config.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "config"), dst="./config", dirs_exist_ok=True)
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
     
    def save_checkpoint(self, epoch):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Epoch number for intermediate checkpoints.
        """
        checkpoint_path = self.ckpt_dir / f'{self.cfg.model_name}_epoch{epoch}.pt'

        torch.save(self.SNN.state_dict(), checkpoint_path)
        
    def save_loss_history(self, loss_hist):
        """
        Save loss history.
        
        Args:
            loss_hist (torch.Tensor): Loss history tensor
            model_name (str): Name of the model
        """
        torch.save(loss_hist, self.ckpt_dir / f'{self.cfg.model_name}_loss_hist.pt')

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
                # print loss and accuracy
                print(f"Epoch {epoch}, Step {step} \nEchoSpike Loss: {torch.stack(loss_hist[-print_interval//self.batch_size:]).mean(axis=0)}")
                print(f"Acc: {torch.stack(accuracies).mean(axis=0)}")
                accuracies = []
                print(f"Spks: {spks * self.batch_size / print_interval}")  # sparsity ratio
                spks = torch.zeros(len(self.SNN.layers) + 1, device=self.device)
            if epoch >= self.cfg.epochs:
                break
            if step % len(self.train_loader) < self.batch_size and epoch % 20 == 0:
                # save checkpoint
                current_epoch_loss = torch.stack(loss_hist[-20 * len(self.train_loader) // self.batch_size:]).mean().item()
                print(f'epoch loss: {current_epoch_loss}')
                self.save_checkpoint(epoch)

        # Save final model and loss history
        loss_hist_tensor = torch.stack(loss_hist)
        self.save_checkpoint()
        self.save_loss_history(loss_hist_tensor)
        
        checkpoint_path = self.ckpt_dir / f'{self.cfg.model_name}.pt'
        torch.save(self.SNN.state_dict(), checkpoint_path)

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
        spk_history = []
        target_list = []
        losses = []

        bf = 0
        target = [torch.randint(self.test_loader.num_classes, (1,)).item() for _ in range(self.batch_size)]
        for _ in trange(int(len(self.test_loader) / self.batch_size)):
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
        return spk_history, target_list, losses
