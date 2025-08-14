import torch
import torch.nn as nn
import torchvision
import snntorch as snn
from snntorch import utils
from sklearn.mixture import GaussianMixture 
from matplotlib import pyplot as plt
from typing import Callable, Iterable, Union
from src.util.metric import MultiMetric, OutputBase
from src.util.replay_buffers import ReplayBuffer, RippleReplayBuffer, GMMReplayBuffer
from src.util.s4 import S4DTrace
from dataclasses import dataclass
from tqdm import tqdm
import dill
import os 
import numpy as np
from src.util.create_tsne import plot_tsne
from tqdm import trange
from typing import *
from warnings import warn

def leaky_integrate(x:torch.Tensor, beta:float=0.8, noisy:bool=False) -> torch.Tensor:
    """
    input:
        x: (T, ...)
        beta: float
    output:
        li: (T, ..,) same shape as x
    """ 
    T = x.shape[0]
    buff = 0
    li = []
    if noisy:
        x = apply_random_flips(x)
    for t in range(T):
        buff = beta * buff + x[t]
        li.append(buff)
    li = torch.stack(li) # * (1-beta)
    return li

def apply_random_flips(x:torch.Tensor) -> torch.Tensor:
    q_f = 0.05
    q_k = 0.95
    one_mask = (x == 1) * (torch.rand_like(x) < (1- q_k))
    zero_mask = (x == 0) * (torch.rand_like(x) < q_f)
    x[one_mask] = 0
    x[zero_mask] = 1
    return x


def random_roll(x:torch.Tensor) -> torch.Tensor:
    """
    input:
        x: (T, B, N)
    output:
        x: (T, B, N)
    """
    rand = torch.rand(1)
    if rand < 0.1:
        x = x.roll(shifts=1, dims=-1)
    elif rand > 0.9:
        x = x.roll(shifts=-1, dims=-1)
    return x


def augment(x:torch.Tensor, mode:str="geo") -> torch.Tensor:
    shape = x.shape
    if mode == "geo":
        aug = torchvision.transforms.RandomResizedCrop(size=shape[-1], scale=(0.5, 1.0))
    else:
        aug = apply_random_flips
    x = x.view(shape[0] * shape[1], shape[2], shape[3], shape[4])
    x = aug(x)
    x = x.view(*shape)
    return x

class OnlineESPPLayer(nn.Module):
    def __init__(
            self, 
            layer: nn.Module,
            rule:str, 
            c_pos:float=1., 
            c_neg:float=-1., 
            input_thr:float=0.02, 
            use_replay:bool=True, 
            stream_data:bool=False, 
            apply_weight:bool=False, 
            learn_temp:bool=False, 
            gamma:float=0.1,
            temp:float=0.5,
            K:int=1,
        ) -> None:
        super().__init__()
        assert c_pos > 0
        assert c_neg < 0
        if rule != "gated_mi" and rule != "espp":
            raise NotImplementedError("rule must either be \"gated_mi\" or \"espp\". Received " + rule)

        self.rule = rule
        self.layer = layer
        self.c_pos = c_pos
        self.c_neg = c_neg
        self.input_thr = input_thr
        self.apply_weight = apply_weight
        self.K = K
        self.temp = temp
        self.learn_temp = learn_temp
        self.gamma = gamma
        self.device = "cpu"
        if learn_temp:
            self.temp = torch.log(torch.exp(torch.Tensor((self.temp,))) - 1) # inverse softplus
            self.temp = nn.Parameter(self.temp) 

        self.use_replay = use_replay
        self.stream_data = stream_data
        self.beta = self.layer.leaky.beta.item()
        if use_replay:
            period = int(torch.tensor(0.5).log().item() / torch.tensor(self.beta).log().item())
            # self.replay_buffer = ReplayBuffer(size=5, period=period, random=True)
            self.replay_buffer = RippleReplayBuffer(size=20)
            # self.replay_buffer = GMMReplayBuffer(size=5, period=200)
        if hasattr(self.layer, "conv"):
            in_channels = self.layer.conv.out_channels
            out_channels = self.layer.conv.out_channels
            self.W = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        elif hasattr(self.layer, "ff"):
            in_features = self.layer.ff.out_features
            out_features = self.layer.ff.out_features
            self.W = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
            # self.W = nn.Parameter(torch.rand(size=(in_features,)))
            # self.trace = S4DTrace(N=out_features, H=32)


    def forward(self, x: torch.Tensor, y: torch.Tensor, input_sum:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        foward passthrough of the layer with espp extension
        input:
            x: (T, B, C, H, W)
            y: (B,)
            input_sum: (T, B)
        output: 
            spk_rec: (T, B, C', H', W'), spk_echo: (T, B, C', H', W'), ...
        """
        self.beta = self.layer.leaky.beta.item() # if beta is learned, it needs to be updates every time
        T = x.shape[0]
        B = x.shape[1]
        if B != 1 and self.stream_data:
            warn(f"Batch size must be 1 if data is ought to be streamed in training. Found a batch size of {B}. Data is not streamed.")
            self.stream_data = False
        rand_idx = torch.randperm(n=B)

        if not self.stream_data:
            utils.reset(self.layer)  # resets hidden states for all LIF neurons in net
        spk_out, mem_out = self.layer(x) # (T, B, C', H', W')

        spikes = spk_out#.reshape([T, B, -1]) # (T, B, N)
        spk_echo = self.calc_echo(spikes) # (B, N)        
        y_espp = y == y[rand_idx] # 0 or 1
        
        echo_sparcity = (spk_echo == 0).float().mean()

        echo_pos = spk_echo
        if self.use_replay:
            echo_neg = []
            for t in range(T):
                echo_neg.append(self.replay_buffer(echo_pos[t,...]))
            echo_neg = torch.stack(echo_neg, axis=0).to(self.device)
        else:
            echo_neg = spk_echo[:, rand_idx, ...]
        spk_echo = spk_echo.detach()
        echo_pos = echo_pos.detach()
        echo_neg = echo_neg.detach()

        if self.K > 1:
            k = torch.randint(low=1, high=self.K+1, size=(1,)).item()
        else:
            k = 1

        sim_score_pos = self.calc_sim_score(spikes, echo_pos, k)
        sim_score_neg = self.calc_sim_score(spikes, echo_neg, k)#.exp().sum(axis=1, keepdim=True).log()
        if self.rule == "espp":
            loss, loss_pos, loss_neg, update_sparcity = self.calc_espp_loss(sim_score_pos, sim_score_neg, input_sum, k)
        elif self.rule == "gated_mi":
            loss, loss_pos, loss_neg, update_sparcity = self.calc_mi_loss(sim_score_pos, sim_score_neg, input_sum, k)

        loss = loss
        
        # if not self.replay_buffer.training_ready():
        #     loss = loss.detach()

        return spk_out, loss, loss_pos, loss_neg, sim_score_pos, sim_score_neg, spk_echo, update_sparcity, echo_sparcity, y_espp
    

    def calc_sim_score(self, spikes:torch.Tensor, echo:torch.Tensor, k:int=1) -> torch.Tensor:
        if self.learn_temp:
            temp = torch.nn.functional.softplus(self.temp)
        else:
            temp = self.temp

        if self.apply_weight:
            spikes = torch.vmap(self.W)(spikes)

        # safe_norm = lambda x: x / self.calc_norm_safe(x, keepdim=True) 
        sim_score: torch.Tensor = spikes[k:,...] * echo[:-k,...]

        if len(spikes.shape) == 3:
            # (T, B, N)
            sim_score = sim_score.mean(axis=(-1,))
        elif len(spikes.shape) == 5:
            # (T, B, C, H, W)
            sim_score = sim_score.mean(axis=(-3, -2, -1))
        else:
            raise NotImplementedError(f"spikes and echo should have 3 or 5 axes for linear and conv, respectively, got {len(spikes.shape)} and {len(echo.shape)}")

        sim_score = sim_score / temp
        return sim_score


    def calc_norm_safe(self, x:torch.Tensor, keepdim:bool) -> torch.Tensor:
        length = x.detach().norm(dim=-1, keepdim=keepdim)
        length[length==0] += 1
        return length

    
    def calc_echo(self, x:torch.Tensor) -> torch.Tensor:
        """
        randomly sample a sequence from within the batch as "echo" sample
        input:
            x: (T, B, C, H, W)
            rand_idx: (B,)
        output: 
            x_echo: (T, B, C, H, W)
        """
        T = x.shape[0]
        B = x.shape[1]
        if hasattr(self, "trace"):
            x_echo = self.trace(x.detach())
        else:
            x_echo = leaky_integrate(x.detach(), self.beta, noisy=False)
        # x_echo = x_echo.sum(axis=0).detach() # (B, C, H, W)
        # n_tot = x_echo.sum(axis=(1, 2, 3)) # (B)
        # for i in range(n_tot.shape[0]):
        #     if n_tot[i] > 0:
        #         x_echo[i] = x_echo[i] / n_tot[i, None, None, None] # ((B), C, H, W)
        # for i in range(B):
        #     for t in range(T):
        #         length = x_echo[t, i].norm()
        #         if length > 0:
        #             x_echo[t, i] = x_echo[t, i] / length
        return x_echo

    def calc_mi_loss(self, sim_score_pos: torch.Tensor, sim_score_neg: torch.Tensor, input_sum: torch.Tensor, k:int=1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input:
            sim_score_pos: (T, B)
            sim_score_neg: (T, B)
            input_sum: (T, B)
        output: 
            loss: ()
            loss_pos: ()
            loss_neg: ()
            update_sparcity: ()
        """
        loss_pos = sim_score_pos
        loss_neg = sim_score_neg

        loss = self.gamma * input_sum[k:,...] - loss_pos + loss_neg

        thr = self.input_thr
        loss = torch.where(loss > 0, loss, 0) # (T, B)
        loss = torch.where(input_sum[k:,...] > thr, loss, 0) # (T, B)

        update_sparcity = (loss == 0).float().mean()
        loss = loss.sum(dim=0).mean()
        loss_pos = loss_pos.sum(dim=0).mean()
        loss_neg = loss_neg.sum(dim=0).mean()

        return loss, loss_pos, loss_neg, update_sparcity
    
    def calc_espp_loss(self, sim_score_pos: torch.Tensor, sim_score_neg: torch.Tensor, input_sum: torch.Tensor, k:int=1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input:
            sim_score_pos: (T, B)
            sim_score_neg: (T, B)
            input_sum: (T, B)
        output: 
            loss: ()
            loss_pos: ()
            loss_neg: ()
            update_sparcity: ()
        """
        loss_pos = self.c_pos * input_sum[k:,...] - sim_score_pos # (T, B)
        loss_neg = self.c_neg * input_sum[k:,...] + sim_score_neg # (T, B)
        
        # max
        loss_pos = torch.where(loss_pos > 0, loss_pos, 0) 
        loss_neg = torch.where(loss_neg > 0, loss_neg, 0) 
        
        # input threshold
        loss_pos = torch.where(input_sum[k:,...] > self.input_thr, loss_pos, 0) # (T, B)
        loss_neg = torch.where(input_sum[k:,...] > self.input_thr, loss_neg, 0) # (T, B)
        
        # reduction
        update_sparcity = ((loss_pos + loss_neg) == 0).float().mean()
        loss_pos = loss_pos.sum(dim=0).mean() # ()
        loss_neg = loss_neg.sum(dim=0).mean() # ()
        loss = loss_pos + loss_neg # ()
        return loss, loss_pos, loss_neg, update_sparcity
    
    def reset(self) -> None:
        if self.use_replay:
            self.replay_buffer.reset()
    
    def to(self, device:str) -> "OnlineESPPLayer":
        self.device = device
        return super().to(device)
    

@dataclass
class ESPPOutput:
    loss: float
    loss_pos: float
    loss_neg: float
    # acc: float
    # acc_pos: float
    # acc_neg: float
    spikes_sum: float
    spikes_mean: float
    sim_score_pos_sum: float
    sim_score_neg_sum: float
    echo_sum: float
    echo_mean: float
    update_sparcity: float
    echo_sparcity: float
    beta: float
    temp: float
    

class OnlineESPPLayerTrainer:
    def __init__(
            self, 
            espp_layer: OnlineESPPLayer, 
            lr: float, 
            train_metrics: MultiMetric,
            val_metrics: MultiMetric,
            test_metrics: MultiMetric,
        ) -> None:
        self.espp_layer = espp_layer
        self.optimizer = torch.optim.SGD(espp_layer.parameters(), lr)

        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics


    def step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, ESPPOutput]:
        """
        input:
            x: (T, B, C, H, W)
            y: (B,)
            input_sum: (T, B)
        output:
            spk_rec: (T, B, C', H', W')
            loss: ()
            output: ESPPOutput
        """
        T = x.shape[0]
        B = x.shape[1]
        spk_rec, loss, loss_pos, loss_neg, sim_score_pos, sim_score_neg, spk_echo, update_sparcity, echo_sparcity, y_espp = self.espp_layer(x, y, input_sum)

        beta = self.espp_layer.beta
        if isinstance(self.espp_layer.temp, nn.Parameter):
            temp = torch.nn.functional.softplus(self.espp_layer.temp.detach()).item()
        else:
            temp = self.espp_layer.temp

        output = ESPPOutput(
            loss=loss.item(), 
            loss_pos=loss_pos.item(), 
            loss_neg=loss_neg.item(), 
            spikes_sum=spk_rec.reshape(T, B, -1).sum(dim=(0, 2)).mean().item(), 
            spikes_mean=spk_rec.mean().item(),
            sim_score_pos_sum=sim_score_pos.sum(dim=0).mean().item(),
            sim_score_neg_sum=sim_score_neg.sum(dim=0).mean().item(),
            echo_sum=spk_echo.reshape(T, B, -1).sum(dim=(0, 2)).mean().item(),
            echo_mean=spk_echo.mean().item(),
            update_sparcity=update_sparcity.item(),
            echo_sparcity=echo_sparcity.item(),
            beta=beta,
            temp=temp,
        )

        return spk_rec, loss, output

    def train_step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor, train:bool=True) -> tuple[torch.Tensor, float]:
        """
        input:
            x: (T, B, C, H, W)
            y: (B,)
            input_sum: (T, B)
        output: 
            x: (T, B, C', H', W')
            loss: float
        """
        self.espp_layer.train()
        x, loss, output = self.step(x, y, input_sum)

        if train and self.espp_layer.use_replay and self.espp_layer.replay_buffer.training_ready():
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
    
        self.train_metrics.accumulate(output)

        return x, loss.item()
    
    def val_step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: (T, B, C, H, W)
            y: (B,)
            input_sum: (T, B)
        output: 
            x: (T, B, C', H', W')
            loss: float
        """
        self.espp_layer.eval()
        x, loss, output = self.step(x, y, input_sum)
        self.val_metrics.accumulate(output)
        return x, loss.item()
    
    def test_step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: (T, B, C, H, W)
            y: (B,)
            input_sum: (T, B)
        output: 
            x: (T, B, C', H', W')
            loss: float
        """
        self.espp_layer.eval()
        x, loss, output = self.step(x, y, input_sum)
        self.test_metrics.accumulate(output)
        return x, loss.item()
    
    def compute_metrics(self) -> None:
        self.train_metrics.compute()
        self.val_metrics.compute()
        self.test_metrics.compute()

    def reset(self) -> None:
        self.espp_layer.reset()

    def to(self, device:str) -> None:
        self.espp_layer = self.espp_layer.to(device)


class OnlineESPPTrainer:
    def __init__(
            self, 
            train_loader: Iterable, 
            val_loader: Iterable, 
            test_loader: Iterable,
            path:Union[str, None],
        ) -> None:
        self.layer_trainers: list[OnlineESPPLayerTrainer] = []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.path = path
        self.device = "cpu"

    def __getitem__(self, idx: int) -> OnlineESPPLayerTrainer:
        return self.layer_trainers[idx]

    def add_layer_trainer(self, layer_trainer: OnlineESPPLayerTrainer) -> None:
        self.layer_trainers.append(layer_trainer)

    def compute_metrics(self) -> None:
        for layer_trainer in self.layer_trainers:
            layer_trainer.compute_metrics()

    def reset(self) -> None:
        for layer_trainer in self.layer_trainers:
            layer_trainer.reset()

    def train(self, epochs: int, layer_in_pbar: int = 0, save:bool=True, log_offset:int=0, layer_to_train:Union[int, None]=None, train_lower_layers_too:bool=False, train_tsne:bool=False) -> tuple[torch.Tensor, list[float]]:
        """
        controlls all training, from running each batch through the layers, updating the layers independently, tracking each layers' metrics, and creating tsne plots
        """
        assert len(self.layer_trainers) > 0
        print(f"start training for {epochs} epochs...")
        pbar = tqdm(range(log_offset, epochs+log_offset, 1))
        for epoch in pbar:
            create_tsne = (self.path is not None) and ((epoch % 5) == 0 or (epoch + 1) == (epochs + log_offset))
            if create_tsne:
                xs = [list() for _ in self.layer_trainers]
                ys = [list() for _ in self.layer_trainers]
            
            self.reset()
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                # x = random_roll(x)
                T = x.shape[0]
                B = x.shape[1]
                input_sum = x.detach()
                input_sum = input_sum.reshape(T, B, -1)
                input_sum = input_sum.mean(dim=-1)
                # input_sum = leaky_integrate(input_sum)
                for j, layer_trainer in enumerate(self.layer_trainers):
                    x = x.detach()
                    train_flag = (layer_to_train is None) or (layer_to_train == j) or (layer_to_train >= j and train_lower_layers_too)
                    # train_flag = False
                    # if train_flag:
                    #     print(j)
                    x, loss = layer_trainer.train_step(x, y, input_sum, train=train_flag) # (T, B, C', H', W')

                    if create_tsne and train_tsne:
                        xs[j].append(x.mean(dim=0).flatten(start_dim=1).detach().cpu().numpy())  
                        ys[j].append(y.detach().cpu().numpy())

                pbar_dict = self.layer_trainers[layer_in_pbar].train_metrics.get_mean_buffer_values()
                pbar.set_postfix(batch=f"{i+1}/{len(self.train_loader)}", **pbar_dict)

            with torch.no_grad():
                self.reset()
                for i, (x, y) in enumerate(self.val_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    T = x.shape[0]
                    B = x.shape[1]
                    input_sum = x.detach()
                    input_sum = input_sum.reshape(T, B, -1)
                    input_sum = input_sum.mean(dim=-1)
                    # input_sum = leaky_integrate(input_sum)
                    for layer_trainer in self.layer_trainers:
                        x = x.detach()
                        x, loss = layer_trainer.val_step(x, y, input_sum) # (T, B, C', H', W')

                
                    pbar_dict = self.layer_trainers[layer_in_pbar].val_metrics.get_mean_buffer_values()
                    pbar.set_postfix(batch=f"{i+1}/{len(self.val_loader)}", **pbar_dict)

            
                self.reset()
                for i, (x, y) in enumerate(self.test_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    T = x.shape[0]
                    B = x.shape[1]
                    # input_sum = leaky_integrate(input_sum)
                    input_sum = x.detach()
                    input_sum = input_sum.reshape(T, B, -1)
                    input_sum = input_sum.mean(dim=-1)
                    for j, layer_trainer in enumerate(self.layer_trainers):
                        x = x.detach()
                        x, loss = layer_trainer.test_step(x, y, input_sum) # (T, B, C', H', W')
                
                        if create_tsne and not train_tsne:
                            xs[j].append(x.mean(dim=0).flatten(start_dim=1).detach().cpu().numpy())  
                            ys[j].append(y.detach().cpu().numpy())
        
                    pbar_dict = self.layer_trainers[layer_in_pbar].test_metrics.get_mean_buffer_values()
                    pbar.set_postfix(batch=f"{i+1}/{len(self.test_loader)}", **pbar_dict)

            self.compute_metrics()
            
            if create_tsne:
                try:
                    for j in range(len(self.layer_trainers)):
                        print(f"creating t-sne plots for layer {j+1}...")
                        embeds = np.concatenate(xs[j], axis=0)
                        labels = np.concatenate(ys[j], axis=0)
                        print(f"{j} std: {embeds.std()}")
                        if not np.allclose(embeds, 0.0):
                            fig, ax = plot_tsne(embeds, labels)
                            fig_path = self.path + "/plots"
                            os.makedirs(fig_path, exist_ok=True)
                            fig.savefig(fig_path + f"/epoch_{epoch + 1}_layer_{j+1}_tsne.png")
                            plt.close(fig)
                except:
                    ...
        
            if save:
                self.save_training(self.path + "final_checkpoint.pkl")
        
        self.save_training(self.path + "final_checkpoint.pkl") # second time so that it gets triggered even if when the for loop doesn't 
    
    def to(self, device:str) -> "OnlineESPPTrainer":
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

    def load(self, checkpoint: dict) -> "OnlineESPPTrainer":
        for i, layer_trainer in enumerate(self.layer_trainers):
            layer_trainer.espp_layer.load_state_dict(checkpoint[f"layer_{i+1}"]["state_dict"])
            layer_trainer.train_metrics = checkpoint[f"layer_{i+1}"]["train_metrics"]
            layer_trainer.val_metrics = checkpoint[f"layer_{i+1}"]["val_metrics"]
            layer_trainer.test_metrics = checkpoint[f"layer_{i+1}"]["test_metrics"]



    

# from torchviz import make_dot
# x=torch.ones(2, requires_grad=True)
# y=2*x
# z=3+x.detach()
# r=(y+z).sum()    
# make_dot(r)