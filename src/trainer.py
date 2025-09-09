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
        
        # Create optimizer
        self.optimizer = torch.optim.SGD([{"params": layer.fc.parameters(), 'lr': self.lr} for layer in self.SNN.layers])
        self.optimizer.zero_grad()
    
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
     
    def save_checkpoint(self, epoch, step=None):
        """
        Save model checkpoint.

        Args:
            epoch (int): Epoch number for intermediate checkpoints.
            step (int, optional): Current training step
        """
        checkpoint_path = self.ckpt_dir / f'{self.cfg.model_name}_epoch_{epoch}.pt'
        checkpoint = {
            'model_state_dict': self.SNN.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'model_name': self.cfg.model_name,
        }
        
        if step is not None:
            checkpoint['step'] = step
            
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
    
    def resume_from_checkpoint(self):
        """
        Resume training from the latest checkpoint.
        
        Returns:
            tuple: (start_epoch, loss_history) where start_epoch is the epoch to resume from
                   and loss_history is the previous loss history
        """
        if not self.resume:
            return 0, []
            
        # Find the latest checkpoint
        checkpoint_files = list(self.ckpt_dir.glob(f'{self.cfg.model_name}_epoch_*.pt'))
        if not checkpoint_files:
            self.logger.warning("No checkpoint files found. Starting from scratch.")
            return 0, []
        
        # Sort by epoch number and get the latest
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        
        self.logger.info(f"Loading checkpoint from: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        
        # Load model state
        self.SNN.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("Loaded optimizer state from checkpoint")
            except Exception as e:
                self.logger.warning(f"Could not load optimizer state: {e}")
        
        # Load loss history if available
        loss_hist_path = self.ckpt_dir / f'{self.cfg.model_name}_loss_hist.pt'
        loss_history = []
        if loss_hist_path.exists():
            try:
                loss_history = torch.load(loss_hist_path, map_location=self.device)
                loss_history = loss_history.tolist() if isinstance(loss_history, torch.Tensor) else loss_history
                self.logger.info(f"Loaded loss history with {len(loss_history)} entries")
            except Exception as e:
                self.logger.warning(f"Could not load loss history: {e}")
                loss_history = []
        
        self.logger.info(f"Resumed from epoch {start_epoch}")
        return start_epoch, loss_history

    def get_latest_checkpoint_path(self):
        """
        Get the path to the latest checkpoint file.
        
        Returns:
            Path or None: Path to the latest checkpoint file, or None if no checkpoints exist
        """
        checkpoint_files = list(self.ckpt_dir.glob(f'{self.cfg.model_name}_epoch_*.pt'))
        if not checkpoint_files:
            return None
        
        # Sort by epoch number and get the latest
        return max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
    
    def load_model_for_inference(self, checkpoint_path=None, load_optimizer=False):
        """
        Load a trained model for inference.
        
        Args:
            checkpoint_path (str or Path, optional): Path to specific checkpoint. 
                                                   If None, loads the latest checkpoint.
            load_optimizer (bool): Whether to load optimizer state for potential training resumption
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint_path()
        
        if checkpoint_path is None:
            self.logger.error("No checkpoint found for inference")
            return False
        
        try:
            self.logger.info(f"Loading model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.SNN.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if requested and available
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.logger.info("Loaded optimizer state for potential training resumption")
                except Exception as e:
                    self.logger.warning(f"Could not load optimizer state: {e}")
            
            self.SNN.eval()
            self.logger.info("Model loaded successfully for inference")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def train(self, freeze=[]):
        """
        Trains a SNN.

        Args:
            freeze (list): List of layers to freeze during training.

        Returns:
            torch.Tensor: A tensor containing the loss history during training.
        """
        torch.set_grad_enabled(False)
        
        # Resume from checkpoint if requested
        start_epoch, loss_hist = self.resume_from_checkpoint()
        
        accuracies = []
        print_interval = 100 * self.batch_size if 'mnist' in self.cfg.dataset else 40 * self.batch_size
        
        self.logger.info(f"Starting training for {self.cfg.epochs} epochs (resuming from epoch {start_epoch})")
        self.logger.info(f"Batch size: {self.batch_size}, Learning rate: {self.lr}")
        self.logger.info(f"Online mode: {self.online}, Augmentation: {self.augment}")
        
        self.SNN.train()
        
        # Initialize training state
        bf = 0
        target = [torch.randint(self.train_loader.num_classes, (1,)).item() for _ in range(self.batch_size)]
        spks = torch.zeros(len(self.SNN.layers) + 1, device=self.device)
        
        # Calculate total steps and current position
        total_steps_per_epoch = len(self.train_loader)
        target_total_steps = self.cfg.epochs * total_steps_per_epoch
        current_step = len(loss_hist) * self.batch_size if loss_hist else 0
        
        self.logger.info(f"Target total steps: {target_total_steps}, Starting from step: {current_step}")
        
        # Main training loop
        while current_step < target_total_steps:
            # Process one batch
            data, target = self.train_loader.next_item(target, contrastive=(bf == -1))
            data = data.float().to(self.device)
            
            if self.augment:
                data = augment_shd(data)

            target = target.to(self.device)
            sample_loss = torch.zeros(len(self.SNN.layers), device=self.device)

            # Process time steps
            for time_step in range(data.shape[0]):
                # Get input activity if online mode
                inp_activity = data[time_step].mean(axis=-1) if self.online else None
                
                # Forward pass
                spk, _, loss, grad = self.SNN(data[time_step], torch.tensor(bf, device=self.device), 
                                            freeze, inp_activity=inp_activity)
                
                # Accumulate spike statistics
                spks += torch.stack([data[time_step].mean(), *[sp.mean() for sp in spk]])
                sample_loss += loss
                
                # Online weight update
                if self.online:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Store loss and accuracy
            loss_hist.append(sample_loss / data.shape[0])
            accuracies.append(self.SNN.reset(bf))

            # Offline weight update (after contrastive batch)
            if bf == -1 and not self.online:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Toggle between predictive and contrastive phases
            bf = 1 if bf != 1 else -1
            
            # Update step counter
            current_step = len(loss_hist) * self.batch_size
            current_epoch = current_step // total_steps_per_epoch
            
            # Logging and checkpointing
            self._handle_logging_and_checkpointing(
                loss_hist, accuracies, spks, current_step, current_epoch, 
                print_interval, total_steps_per_epoch
            )
            
            # Reset spike counter and accuracies after logging
            if current_step % print_interval < self.batch_size and len(loss_hist) > 1:
                accuracies = []
                spks = torch.zeros(len(self.SNN.layers) + 1, device=self.device)

        # Save final model and loss history
        loss_hist_tensor = torch.stack(loss_hist)
        self.save_loss_history(loss_hist_tensor)
        self.save_checkpoint(current_epoch, current_step)
        
        self.logger.info(f"Training completed. Total epochs: {current_epoch}")
        return loss_hist_tensor
    
    def _handle_logging_and_checkpointing(self, loss_hist, accuracies, spks, current_step, 
                                        current_epoch, print_interval, total_steps_per_epoch):
        """
        Handle logging and checkpointing during training.
        
        Args:
            loss_hist (list): List of loss values
            accuracies (list): List of accuracy values
            spks (torch.Tensor): Spike statistics
            current_step (int): Current training step
            current_epoch (int): Current epoch
            print_interval (int): Interval for logging
            total_steps_per_epoch (int): Total steps per epoch
        """
        # Periodic logging
        if current_step % print_interval < self.batch_size and len(loss_hist) > 1:
            avg_loss = torch.stack(loss_hist[-print_interval//self.batch_size:]).mean(axis=0)
            avg_acc = torch.stack(accuracies).mean(axis=0)
            spikes = spks * self.batch_size / print_interval
            
            self.logger.info(f"Epoch {current_epoch}, Step {current_step}")
            self.logger.info(f"EchoSpike Loss: {avg_loss}")
            self.logger.info(f"Accuracy: {avg_acc}")
            self.logger.info(f"Spikes: {spikes}")
        
        # Periodic checkpointing (every 20 epochs)
        if (current_step % total_steps_per_epoch < self.batch_size and 
            current_epoch % 20 == 0 and current_epoch > 0):
            recent_losses = loss_hist[-20 * total_steps_per_epoch // self.batch_size:]
            if recent_losses:
                current_epoch_loss = torch.stack(recent_losses).mean().item()
                self.logger.info(f'Epoch {current_epoch} loss: {current_epoch_loss}')
                self.save_checkpoint(current_epoch, current_step)

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
