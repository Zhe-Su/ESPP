import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import numpy as np

from typing import Callable, Iterable
from src.util.metric import MultiMetric
from dataclasses import dataclass
from tqdm import tqdm, trange
import dill
from typing import Union, Callable
from src.util.create_tsne import plot_tsne, tsne_ready_torch_to_numpy
import os

class InfoNCELayer(nn.Module):
    """
    wraps a layer with the info nce loss
    """
    def __init__(
            self,
            layer:nn.Module,
            in_channels:int, 
            out_channels:int, 
            beta:float=0.5, 
            spike_grad:Callable=surrogate.atan(), 
            k:int=4, 
            leaky_enc:bool=True, 
            spike_context:bool=True,
            apply_weight:bool=True,
        ) -> None:
        super().__init__()
        self.layer = layer

        self.weight = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.leaky = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        self.k = k
        self.leaky_enc = leaky_enc
        self.spike_context = spike_context
        self.apply_weight = apply_weight
    
    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor ,torch.Tensor]:
        spk_rec, mem_rec = self.layer(x)
        context = self.calc_context(spk_rec, mem_rec)
        loss, log_score_pos, log_score_neg = self.calc_infonce_loss(spk_rec, context)
        return spk_rec, loss, log_score_pos, log_score_neg

    def calc_infonce_loss(self, spk_rec:torch.Tensor, context:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input:
            spk_rec: (T, B, C, H, W)
            context: (T, B, C, H, W)
        output:
            loss: ()
            log_score_pos: ()
            log_score_neg: ()
        """
        T = spk_rec.shape[0]
        B = spk_rec.shape[1]
        C = spk_rec.shape[2]
        H = spk_rec.shape[3]
        W = spk_rec.shape[4]
        if self.apply_weight:
            x_pos = torch.vmap(lambda x: self.weight(x))(spk_rec)
        else:
            x_pos = spk_rec
        rand_idx = torch.randperm(n=B)
        x_neg = x_pos[:, rand_idx, ...]

        # (T, B)
        score_pos = (context[:-self.k, ...] * x_pos[self.k:, ...]).mean(dim=[2, 3, 4]).exp()
        score_neg = (context[:-self.k, ...] * x_neg[self.k:, ...]).mean(dim=[2, 3, 4]).exp()

        score_pos = score_pos + 1e-4
        score_neg = score_neg + 1e-4

        log_score_pos = score_pos.log()
        log_score_neg = score_neg.sum(dim=1, keepdim=True).log()

        loss = - log_score_pos + log_score_neg
        loss = loss.mean(dim=0).mean(dim=0)
        return loss, log_score_pos, log_score_neg 
    
    def calc_context(self, spk_rec:torch.Tensor, mem_rec:torch.Tensor) -> torch.Tensor:
        """
        input:
            spk_rec: (T, B, C, H, W)
            mem_rec: (T, B, C, H, W)
        output:
            context: (T, B, C, H, W)
        """
        T = spk_rec.shape[0]
        if self.leaky_enc:
            utils.reset(self.leaky)
            context_rec = []
            for t in range(T):
                spk_out, mem_out = self.leaky(spk_rec[t])
                if self.spike_context:
                    context_rec.append(spk_out)
                else:
                    context_rec.append(mem_out)
            context = torch.stack(context_rec)
        else:
            if self.spike_context:
                context = spk_rec
            else:
                context = mem_rec

        if context is None:
            raise RuntimeError("Context is None. Perhaps the network is not setup appropriately")
        return context
    
@dataclass
class InfoNCEOutput:
    loss: float
    log_score_pos: float
    log_score_neg: float
    spikes_sum: float
    spikes_mean: float
    

class InfoNCELayerTrainer:
    """
    contains all the necessary functionalities to train a single info nce layer
    """
    def __init__(
            self,
            info_nce_layer: InfoNCELayer,
            lr: float,
            train_metrics: MultiMetric,
            val_metrics: MultiMetric,
            test_metrics: MultiMetric,
        ) -> None:
        self.info_nce_layer = info_nce_layer
        self.optimizer = torch.optim.AdamW(self.info_nce_layer.parameters(), lr, betas=(0.9, 0.999))
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
    
    def step(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, InfoNCEOutput]:
        """
        input:
            x: (T, B, C, H, W)
        output:
            spk_rec: (T, B, C', H', W')
            loss: ()
            output: InfoNCEOutput
        """
        spk_rec, loss, log_score_pos, log_score_neg = self.info_nce_layer(x)

        output = InfoNCEOutput(
            loss=loss.item(), 
            spikes_sum=spk_rec.sum().item(), 
            spikes_mean=spk_rec.mean(dim=1).sum().item(), 
            log_score_pos=log_score_pos.mean().item(), 
            log_score_neg=log_score_neg.mean().item()
        )

        return spk_rec, loss, output
    
    def train_step(self, x:torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        input:
            x: (T, B, C, H, W)
        output:
            x: (T, B, C', H', W')
            loss: float
        """
        self.info_nce_layer.train()
        x, loss, output = self.step(x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_metrics.accumulate(output)
        return x, loss.item()
    
    def val_step(self, x:torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        input:
            x: (T, B, C, H, W)
        output:
            x: (T, B, C', H', W')
            loss: float
        """
        self.info_nce_layer.eval()
        x, loss, output = self.step(x)
        self.val_metrics.accumulate(output)
        return x, loss.item()
    
    def test_step(self, x:torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        input:
            x: (T, B, C, H, W)
        output:
            x: (T, B, C', H', W')
            loss: float
        """
        self.info_nce_layer.eval()
        x, loss, output = self.step(x)
        self.test_metrics.accumulate(output)
        return x, loss.item()
        
    def compute_metrics(self) -> None:
        self.train_metrics.compute()
        self.val_metrics.compute()
        self.test_metrics.compute()

    def to(self, device:str) -> None:
        self.info_nce_layer = self.info_nce_layer.to(device)


class InfoNCETrainer:
    def __init__(
            self, 
            train_loader:Iterable,  
            val_loader:Iterable, 
            test_loader:Iterable,
            path:Union[str, None],
        ) -> None:
        self.layer_trainers: list[InfoNCELayerTrainer] = []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = "cpu"
        self.path = path

    def __getitem__(self, idx:int) -> InfoNCELayerTrainer:
        return self.layer_trainers[idx]
    
    def add_layer_trainer(self, layer_trainer: InfoNCELayerTrainer) -> None:
        self.layer_trainers.append(layer_trainer)

    def compute_metrics(self) -> None:
        for layer_trainer in self.layer_trainers:
            layer_trainer.compute_metrics()

    def train(self, epochs: int, layer_in_pbar: int = 0) -> tuple[torch.Tensor, list[float]]:
        """
        controlls all training, from running each batch through the layers, updating the layers independently, tracking each layers' metrics, and creating tsne plots
        """
        assert len(self.layer_trainers) > 0

        pbar = trange(epochs)
        for epoch in pbar:
            create_tsne = (self.path is not None) and ((epoch % 5) == 0 or (epoch+1) == epochs)
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                for layer_trainer in self.layer_trainers:
                    x = x.detach()
                    x, loss = layer_trainer.train_step(x) # (T, B, C', H', W')
                
                pbar_dict = self.layer_trainers[layer_in_pbar].train_metrics.get_mean_buffer_values()
                pbar.set_postfix(batch=f"{i+1}/{len(self.train_loader)}", **pbar_dict)

            with torch.no_grad():
                for i, (x, y) in enumerate(self.val_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    for layer_trainer in self.layer_trainers:
                        x = x.detach()
                        x, loss = layer_trainer.val_step(x) # (T, B, C', H', W')
                
                    pbar_dict = self.layer_trainers[layer_in_pbar].val_metrics.get_mean_buffer_values()
                    pbar.set_postfix(batch=f"{i+1}/{len(self.val_loader)}", **pbar_dict)

                if create_tsne:
                    xs = [list() for _ in self.layer_trainers]
                    ys = [list() for _ in self.layer_trainers]
            
                for i, (x, y) in enumerate(self.test_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    for j, layer_trainer in enumerate(self.layer_trainers):
                        x = x.detach()
                        x, loss = layer_trainer.test_step(x) # (T, B, C', H', W')
                
                        if create_tsne:
                            xs[j].append(x.mean(dim=0).flatten(start_dim=1).detach().cpu().numpy())  
                            ys[j].append(y.detach().cpu().numpy())
        
                    pbar_dict = self.layer_trainers[layer_in_pbar].test_metrics.get_mean_buffer_values()
                    pbar.set_postfix(batch=f"{i+1}/{len(self.test_loader)}", **pbar_dict)

            self.compute_metrics()
            
            if create_tsne:
                for j in range(len(self.layer_trainers)):
                    print(f"creating t-sne plots for layer {j+1}...")
                    embeds = np.concatenate(xs[j], axis=0)
                    labels = np.concatenate(ys[j], axis=0)
                    fig, ax = plot_tsne(embeds, labels)
                    fig_path = self.path + "/plots"
                    os.makedirs(fig_path, exist_ok=True)
                    fig.savefig(fig_path + f"/epoch_{epoch+1}_layer_{j+1}_tsne.png")
        
        self.save_training(self.path + "final_checkpoint.pkl")

    def to(self, device:str) -> "InfoNCETrainer":
        for layer_trainer in self.layer_trainers:
            layer_trainer.to(device)
        self.device = device
        return self
    
    def save_training(self, path: str) -> None:
        checkpoint = {}
        for i, layer_trainer in enumerate(self.layer_trainers):
            checkpoint[f"layer_{i+1}"] = {}
            checkpoint[f"layer_{i+1}"]["state_dict"] = layer_trainer.info_nce_layer.state_dict()
            checkpoint[f"layer_{i+1}"]["train_metrics"] = layer_trainer.train_metrics
            checkpoint[f"layer_{i+1}"]["val_metrics"] = layer_trainer.val_metrics
            checkpoint[f"layer_{i+1}"]["test_metrics"] = layer_trainer.test_metrics
        dill.dump(checkpoint, open(path, "wb"))

    @classmethod
    def load(cls, path: str) -> "InfoNCETrainer":
        """
        DEPRICATED,
        TODO: FIX
        """
        return dill.load(open(path, "rb"))