import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from typing import Callable, Iterable, Union
from src.util.metric import MultiMetric, OutputBase
from dataclasses import dataclass
from tqdm import tqdm
import dill
import os 
import numpy as np
from src.util.create_tsne import plot_tsne
from tqdm import trange

class ESPPLayer(nn.Module):
    def __init__(self, layer: nn.Module, c_pos:float=1., c_neg:float=-1., input_thr:float=0.02) -> None:
        super().__init__()
        assert c_pos > 0
        assert c_neg < 0
        self.layer = layer
        self.c_pos = c_pos
        self.c_neg = c_neg
        self.input_thr = input_thr
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, input_sum:torch.Tensor, max_spikes:int, contrastive:bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        foward passthrough of the layer with espp extension
        input:
            x: (T, B, C, H, W)
            y: (B,)
            input_sum: (T, B)
            max_spikes: int
        output: 
            spk_rec: (T, B, C', H', W'), x_echo: (B, C', H', W')
        """
        B = x.shape[1]
        rand_idx = torch.randperm(n=B)

        utils.reset(self.layer)  # resets hidden states for all LIF neurons in net
        spk_rec, mem_rec = self.layer(x) # (T, B, C, H, W)
        spk_echo = self.calc_echo(spk_rec, rand_idx) # (B, C, H, W)
        
        y_espp = y == y[rand_idx] # 0 or 1
        
        spk_echo = spk_echo.detach()

        sim_score = (spk_rec * spk_echo[None,...]) # (T, B, C, H, W)
        sim_score = sim_score.sum(axis=(2, 3, 4)) # (T, B)
        loss, loss_pos, loss_neg, update_sparcity = self.calc_espp_loss(sim_score, y_espp, input_sum, max_spikes, contrastive)
        
        return spk_rec, loss, loss_pos, loss_neg, sim_score, spk_echo, update_sparcity, y_espp
    
    def calc_echo(self, x:torch.Tensor, rand_idx:torch.Tensor) -> torch.Tensor:
        """
        randomly sample a sequence from within the batch as "echo" sample
        input:
            x: (T, B, C, H, W)
            rand_idx: (B,)
        output: 
            x_echo: (B, C, H, W)
        """
        x_echo = x[:, rand_idx, ...].detach()
        x_echo = x_echo.sum(axis=0) # (B, C, H, W)
        n_tot = x_echo.sum(axis=(1, 2, 3)) # (B)
        for i in range(n_tot.shape[0]):
            if n_tot[i] > 0:
                x_echo[i] = x_echo[i] / n_tot[i, None, None, None] # ((B), C, H, W)
        return x_echo
    
    def calc_espp_loss(self, sim_score: torch.Tensor, y_espp: torch.Tensor, input_sum: torch.Tensor, max_spikes:int, contrastive:bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input:
            sim_score: (T, B)
            y_espp: (B,)
            input_sum: (T, B)
            max_spikes: int
        output: 
            loss: ()
            loss_pos: ()
            loss_neg: ()
        """
        # pos_sim_score = y_espp[None, :, None, None, None] * sim_score
        # c(y) - y s*s_prev seperated by y values
        loss_pos = torch.where(y_espp[None, :], self.c_pos * input_sum - sim_score, 0) # (T, B)
        loss_neg = torch.where(~y_espp[None, :], self.c_neg * input_sum + sim_score, 0) # (T, B)
        
        # loss_pos = self.c_pos * input_sum - sim_score
        # loss_neg = self.c_neg * input_sum + sim_score

        # max
        loss_pos = torch.where(loss_pos > 0, loss_pos, 0) 
        loss_neg = torch.where(loss_neg > 0, loss_neg, 0) 
        
        # input threshold
        thr = self.input_thr
        loss_pos = torch.where(input_sum > thr, loss_pos, 0) # (T, B)
        loss_neg = torch.where(input_sum > thr, loss_neg, 0) # (T, B)
        
        # reduction
        update_sparcity = ((loss_pos + loss_neg) == 0).float().mean()  
        loss_pos = loss_pos.sum(dim=0).mean() # ()
        loss_neg = loss_neg.sum(dim=0).mean() # ()
        loss = loss_pos * (not contrastive) + loss_neg * contrastive # ()
        return loss, loss_pos, loss_neg, update_sparcity
     

@dataclass
class ESPPOutput:
    loss: float
    loss_pos: float
    loss_neg: float
    acc: float
    acc_pos: float
    acc_neg: float
    spikes_sum: float
    spikes_mean: float
    sim_score_sum: float
    sim_score_mean: float
    echo_sum: float
    echo_mean: float
    update_sparcity: float
    

class ESPPLayerTrainer:
    def __init__(
            self, 
            espp_layer: ESPPLayer, 
            lr: float, 
            train_metrics: MultiMetric,
            val_metrics: MultiMetric,
            test_metrics: MultiMetric,
        ) -> None:
        self.espp_layer = espp_layer
        self.optimizer = torch.optim.SGD(espp_layer.layer.parameters(), lr)

        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    def calc_acc(self, sim_score: torch.Tensor, y_espp:torch.Tensor, input_sum:torch.Tensor) -> tuple[float, float, float]:
        """
        input:
            sim_score: (T, B)
            y_espp: (B,)
            input_sum: (T, B)
        output: 
            acc: float
            acc_pos: float
            acc_neg: float
        """
        sim_score_pos = torch.where(y_espp[None, :], sim_score, 0) # (T, B)
        sim_score_neg = torch.where(~y_espp[None, :], sim_score, 0) # (T, B)

        acc_pos = (self.espp_layer.c_pos*input_sum - sim_score_pos) <= 0
        acc_neg = (self.espp_layer.c_neg*input_sum + sim_score_neg) <= 0
        acc_pos = acc_pos.sum() / y_espp.sum()
        acc_neg = acc_neg.sum() / ~y_espp.sum()

        acc = (acc_pos + acc_neg) / 2
        return acc.item(), acc_pos.item(), acc_neg.item()

    def step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor, max_spikes:int, contrastive:bool) -> tuple[torch.Tensor, torch.Tensor, ESPPOutput]:
        """
        input:
            x: (T, B, C, H, W)
            y: (B,)
            input_sum: (T, B)
            max_spikes: int
        output:
            spk_rec: (T, B, C', H', W')
            loss: ()
            output: ESPPOutput
        """
        spk_rec, loss, loss_pos, loss_neg, sim_score, spk_echo, update_sparcity, y_espp = self.espp_layer(x, y, input_sum, max_spikes, contrastive)

        acc, acc_pos, acc_neg = self.calc_acc(sim_score, y_espp, input_sum)

        output = ESPPOutput(
            loss=loss.item(), 
            loss_pos=loss_pos.item(), 
            loss_neg=loss_neg.item(), 
            acc=acc, 
            acc_pos=acc_pos, 
            acc_neg=acc_neg, 
            spikes_sum=spk_rec.sum(dim=(0, 2, 3, 4)).mean().item(), 
            spikes_mean=spk_rec.mean().item(),
            sim_score_sum=sim_score.sum(dim=0).mean().item(),
            sim_score_mean=sim_score.mean().item(),
            echo_sum=spk_echo.sum(dim=(1, 2, 3)).mean().item(),
            echo_mean=spk_echo.mean().item(),
            update_sparcity=update_sparcity.item(),
        )

        return spk_rec, loss, output

    def train_step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor, max_spikes:int, contrastive:bool) -> tuple[torch.Tensor, float]:
        """
        input:
            x: (T, B, C, H, W)
            y: (B,)
            input_sum: (T, B)
            max_spikes: int
        output: 
            x: (T, B, C', H', W')
            loss: float
        """
        self.espp_layer.train()
        x, loss, output = self.step(x, y, input_sum, max_spikes, contrastive)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_metrics.accumulate(output)

        return x, loss.item()
    
    def val_step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor, max_spikes:int, contrastive:bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: (T, B, C, H, W)
            y: (B,)
            input_sum: (T, B)
            max_spikes: int
        output: 
            x: (T, B, C', H', W')
            loss: float
        """
        self.espp_layer.eval()
        x, loss, output = self.step(x, y, input_sum, max_spikes, contrastive)
        self.val_metrics.accumulate(output)
        return x, loss.item()
    
    def test_step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor, max_spikes:int, contrastive:bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: (T, B, C, H, W)
            y: (B,)
            input_sum: (T, B)
            max_spikes: int
        output: 
            x: (T, B, C', H', W')
            loss: float
        """
        self.espp_layer.eval()
        x, loss, output = self.step(x, y, input_sum, max_spikes, contrastive)
        self.test_metrics.accumulate(output)
        return x, loss.item()
    
    def compute_metrics(self) -> None:
        self.train_metrics.compute()
        self.val_metrics.compute()
        self.test_metrics.compute()

    def to(self, device:str) -> None:
        self.espp_layer = self.espp_layer.to(device)


class ESPPTrainer:
    def __init__(
            self, 
            train_loader: Iterable, 
            val_loader: Iterable, 
            test_loader: Iterable,
            path:Union[str, None],
        ) -> None:
        self.layer_trainers: list[ESPPLayerTrainer] = []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.path = path
        self.device = "cpu"

    def __getitem__(self, idx: int) -> ESPPLayerTrainer:
        return self.layer_trainers[idx]

    def add_layer_trainer(self, layer_trainer: ESPPLayerTrainer) -> None:
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
                input_sum = x.mean(dim=[2, 3, 4])
                max_spikes = sum(x.shape[-3:])
                for layer_trainer in self.layer_trainers:
                    x = x.detach()
                    x, loss = layer_trainer.train_step(x, y, input_sum, max_spikes, contrastive=(i%2)==0) # (T, B, C', H', W')
                
                pbar_dict = self.layer_trainers[layer_in_pbar].train_metrics.get_mean_buffer_values()
                pbar.set_postfix(batch=f"{i+1}/{len(self.train_loader)}", **pbar_dict)

            with torch.no_grad():
                for i, (x, y) in enumerate(self.val_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    input_sum = x.mean(dim=[2, 3, 4])
                    max_spikes = sum(x.shape[-3:])
                    for layer_trainer in self.layer_trainers:
                        x = x.detach()
                        x, loss = layer_trainer.val_step(x, y, input_sum, max_spikes, contrastive=(i%2)==0) # (T, B, C', H', W')
                
                    pbar_dict = self.layer_trainers[layer_in_pbar].val_metrics.get_mean_buffer_values()
                    pbar.set_postfix(batch=f"{i+1}/{len(self.val_loader)}", **pbar_dict)

                if create_tsne:
                    xs = [list() for _ in self.layer_trainers]
                    ys = [list() for _ in self.layer_trainers]
            
                for i, (x, y) in enumerate(self.test_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    input_sum = x.mean(dim=[2, 3, 4])
                    max_spikes = sum(x.shape[-3:])
                    for j, layer_trainer in enumerate(self.layer_trainers):
                        x = x.detach()
                        x, loss = layer_trainer.test_step(x, y, input_sum, max_spikes, contrastive=(i%2)==0) # (T, B, C', H', W')
                
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
    
    def to(self, device:str) -> "ESPPTrainer":
        for layer_trainer in self.layer_trainers:
            layer_trainer.to(device)
        self.device = device
        return self

    def save_training(self, path: str) -> None:
        checkpoint = {}
        for i, layer_trainer in enumerate(self.layer_trainers):
            checkpoint[f"layer_{i+1}"] = {}
            checkpoint[f"layer_{i+1}"]["state_dict"] = layer_trainer.espp_layer.state_dict()
            checkpoint[f"layer_{i+1}"]["train_metrics"] = layer_trainer.train_metrics
            checkpoint[f"layer_{i+1}"]["val_metrics"] = layer_trainer.val_metrics
            checkpoint[f"layer_{i+1}"]["test_metrics"] = layer_trainer.test_metrics
        dill.dump(checkpoint, open(path, "wb"))

    @classmethod
    def load(cls, path: str) -> "ESPPTrainer":
        """
        DEPRICATED,
        TODO: FIX
        """
        return dill.load(open(path, "rb"))



    

# from torchviz import make_dot
# x=torch.ones(2, requires_grad=True)
# y=2*x
# z=3+x.detach()
# r=(y+z).sum()    
# make_dot(r)