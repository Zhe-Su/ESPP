import pickle
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    
    # Initialize trainer (handles dataset creation, model creation, and directory setup)
    trainer = Trainer(cfg)
    
    # Train model
    trainer.train()


if __name__ == '__main__':
    main()


