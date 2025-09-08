import pickle
import os

import hydra
import torch
from data import load_classwise_NMNIST, load_classwise_PMNIST, load_SHD
from model import EchoSpike
from omegaconf import DictConfig, OmegaConf
from utils import train


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    
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
    
    with open(f'models/{cfg.model_name}_args.pkl', 'wb') as f:
        pickle.dump(OmegaConf.to_container(cfg), f)

    loss_hist = train(SNN, train_loader, cfg.epochs, cfg.device, cfg.model_name,
                            batch_size=cfg.batch_size, online=cfg.online, lr=cfg.lr, augment=cfg.augment)

    # Save the model, loss history and arguments
    torch.save(SNN.state_dict(), f'models/{cfg.model_name}.pt')
    torch.save(loss_hist, f'models/{cfg.model_name}_loss_hist.pt')


if __name__ == '__main__':
    main()


