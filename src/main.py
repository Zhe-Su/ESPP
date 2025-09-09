import pickle
import os
import shutil
from pathlib import Path

import hydra
import torch
from data import load_classwise_NMNIST, load_classwise_PMNIST, load_SHD
from model import EchoSpike
from omegaconf import DictConfig, OmegaConf
from utils import train


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    
    # Create directory structure and copy files if not resuming
    ckpt_dir = Path('checkpoints')
    media_dir = Path('media')

    if not cfg.common.resume:
        config_dir = Path('config')
        config_path = config_dir / 'config.yaml'
        config_dir.mkdir(exist_ok=False, parents=False)
        shutil.copy('.hydra/config.yaml', config_path)
        shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
        shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "config"), dst="./config")
        ckpt_dir.mkdir(exist_ok=False, parents=False)
        media_dir.mkdir(exist_ok=False, parents=False)
    
    # Set c_y based on online mode
    if hasattr(cfg, 'c_y_offline') and not cfg.online:
        cfg.c_y = cfg.c_y_offline
    
    # load dataset
    if cfg.dataset == 'nmnist':
        train_loader, _, test_loader = load_classwise_NMNIST(cfg.n_time_bins, split_train=True, batch_size=cfg.batch_size) 
    elif cfg.dataset == 'pmnist':
        train_loader, test_loader = load_classwise_PMNIST(cfg.n_time_bins, scale=cfg.poisson_scale, batch_size=cfg.batch_size)
    elif cfg.dataset == 'shd':
        train_loader, test_loader = load_SHD(batch_size=cfg.batch_size)

    # train model
    SNN = EchoSpike(cfg.n_inputs, cfg.n_hidden, c_y=cfg.c_y, beta=cfg.beta,
                     device=cfg.device, recurrency_type=cfg.recurrency_type,
                     n_time_steps=cfg.n_time_bins, online=cfg.online, inp_thr=cfg.inp_thr).to(cfg.device)
    SNN.reset(0) # after sending to cuda the espp_layers' mem is not on cuda, resetting fixes that
    
    # Create models directory if it doesn't exist

    loss_hist = train(SNN, train_loader, cfg.epochs, cfg.device, cfg.model_name,
                            batch_size=cfg.batch_size, online=cfg.online, lr=cfg.lr, augment=cfg.augment)

    # Save the model, loss history and arguments
    torch.save(SNN.state_dict(), ckpt_dir / f'{cfg.model_name}.pt')
    torch.save(loss_hist, ckpt_dir / f'{cfg.model_name}_loss_hist.pt')


if __name__ == '__main__':
    main()


