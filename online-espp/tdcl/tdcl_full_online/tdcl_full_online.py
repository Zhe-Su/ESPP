import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from typing import Callable, Iterable, Union
from tdcl.util.metric import MultiMetric, OutputBase
from dataclasses import dataclass
from tqdm import tqdm
import dill
import os 
import numpy as np
from tdcl.util.create_tsne import plot_tsne
from tqdm import trange
from warnings import warn


@torch.jit.script
def calc_M(surr_mem_out:torch.Tensor, in_trace:torch.Tensor) -> torch.Tensor:
    # M = surr_mem_out[:,:,None] * in_trace[:,None,:] # (B, N', N)
    M = torch.einsum("bk, bl -> bkl", surr_mem_out,  in_trace) # (B, N', N)
    return M

@torch.jit.script
def update(dL, M, trace, N:int):
    # M = surr_mem_out[:,:,None] * in_trace[:,None,:] # (B, N', N)
    # M_trace_pos = self.m_trace # (B, N', N)
    # M_trace_neg = self.trace_replay_buffer(M_trace_pos)
    
    # calculate gradients
    # dW_pos = echo_pos[:,:,None] * M # + M_trace_pos * spikes[:,:,None] # (B, N', N)
    dW_pos = torch.einsum("bk, bkl -> bkl", trace, M) # + M_trace_pos * spikes[:,:,None] # (B, N', N)
    # dW_neg = echo_neg[:,:,None] * M # + M_trace_neg * spikes[:,:,None] # (B, N', N)
    dW = -dL[:, None, None].float() * dW_pos
    dW = dW.mean(dim=0) # (N', N)
    dW = dW / N
    # update layer weights
    return dW


@torch.jit.script
def update_with_trace(dL, M, spikes, echo_pos, echo_neg, M_trace_pos, M_trace_neg, N:int):
    # M = surr_mem_out[:,:,None] * in_trace[:,None,:] # (B, N', N)
    dW_pos = echo_pos[:,:,None] * M + M_trace_pos * spikes[:,:,None] # (B, N', N)
    dW_neg = echo_neg[:,:,None] * M + M_trace_neg * spikes[:,:,None] # (B, N', N)
    dW = -dL[:, None, None].float() * (dW_pos - dW_neg)
    dW = dW.mean(dim=0) # (N', N)
    dW = dW / N
    return dW


class FullOnlineTDCLLayer(nn.Module):
    def __init__(
            self, 
            layer: nn.Module, 
            tracer,
            lr: float,
            warm_up:int,
            input_thr:float=0.02,
            gamma:float=0.1,
            temp:float=0.5,
        ) -> None:
        super().__init__()
        self.layer = layer
        self.input_thr = input_thr
        self.gamma = gamma
        self.temp = temp
        self.warm_up = warm_up
        self.counter = 0
        self.tracer = tracer

        self.lr = lr

        self.device = "cpu"

        self.beta = self.layer.leaky.beta

        self.init_traces = True
        self.in_trace = None

    def forward(self, x: torch.Tensor, input_sum:torch.Tensor, train:bool):
        """
        step passthrough of the layer with espp extension
        input:
            x: (B, C, H, W)
            y: (B)
            input_sum: ()
            max_spikes: int
        output: 
            spk_rec: (B, C', H', W'), x_echo: (B, C', H', W')
        """
        # running through layer
        B = x.shape[0]
        spk_out, mem_out = self.layer.step(x) # (B, N)
        spikes = spk_out.reshape([B, -1]) # (B, N')
        N = spikes.shape[-1]
        if self.init_traces:
            self.in_trace = torch.zeros(B, sum(x.shape[1:]), device=self.device)
            for (_, li) in self.tracer.traces:
                li.state = torch.zeros(B, self.layer.get_out_dim(), device=self.device)
            self.init_traces = False
            train = False
        
        # get echos
        trace = self.tracer.get()
        # calc sim scores
        sim_score = spikes * trace # (B, N')
        sim_score = sim_score.mean(axis=(-1)) / self.temp# (B)
        
        # calc loss
        loss, update_sparcity = self.calc_mi_loss(sim_score, input_sum)
        surr_mem_out = self.surr(mem_out-1)# * mem_out
        self.in_trace = self.beta * self.in_trace + x.reshape(B, -1)
        M = calc_M(surr_mem_out, self.in_trace)
        if train and self.counter >= self.warm_up:
            W = self.layer.get_weight()
            dL = (loss > 0) * (input_sum > self.input_thr)
            
            dW = update(dL, M, trace, N)
            
            W = W - (self.lr / self.temp) * dW 
            self.layer.set_weight(W)

        else:
            self.counter = self.counter + 1

        # update traces
        loss = loss.mean()
        self.tracer.step(spikes)

        # some metrics
        trace_sparcity = (trace == 0).float().mean()
        return spk_out, loss, sim_score, trace, update_sparcity, trace_sparcity


    def surr(self, mem:torch.Tensor) -> torch.Tensor:
        # return (1 / torch.pi) * (1 / (1 +  (torch.pi*mem)**2)) 
        return 1 / (1 +  (torch.pi*mem)**2) 
        

    def calc_mi_loss(self, sim_score: torch.Tensor, input_sum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # loss = torch.log(1 + input_sum) - sim_score_pos + sim_score_neg
        loss = self.gamma * input_sum - sim_score 
        loss = torch.where(loss > 0, loss, 0) # (B)
        loss = torch.where(input_sum > self.input_thr, loss, 0) # (B)

        update_sparcity = (loss == 0).float().mean()
        loss = loss#.mean()
        return loss, update_sparcity
    
    def reset(self) -> None:
        utils.reset(self)
        self.in_trace = None
        self.init_traces = True
        self.tracer.reset()
        self.counter = 0

    def to(self, device:str) -> None:
        super().to(device)
        self.beta = self.beta.to(device)
        self.tracer = self.tracer.to(device)
        self.device = device
        return self


@dataclass
class TDCLOutput:
    loss: float
    spikes_sum: float
    spikes_mean: float
    sim_score_sum: float
    trace_sum: float
    trace_mean: float
    update_sparcity: float
    trace_sparcity: float
    

class FullOnlineTDCLLayerTrainer:
    def __init__(
            self, 
            tdcl_layer: FullOnlineTDCLLayer, 
            train_metrics: MultiMetric,
            val_metrics: MultiMetric,
            test_metrics: MultiMetric,
        ) -> None:
        self.tdcl_layer = tdcl_layer

        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    def step(self, x:torch.Tensor, input_sum:torch.Tensor, train:bool) -> tuple[torch.Tensor, torch.Tensor, TDCLOutput]:
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
        T = x.shape[0]
        B = x.shape[1]

        spk_rec_list = []
        loss_list = []
        sim_score_list = []
        trace_list = []
        update_sparcity_list = []
        trace_sparcity_list = []
        for t in range(T):
            spk_rec, loss, sim_score, trace, update_sparcity, trace_sparcity = self.tdcl_layer(x[t,...], input_sum[t,...], train)
            spk_rec_list.append(spk_rec)
            loss_list.append(loss)
            sim_score_list.append(sim_score)
            trace_list.append(trace)
            update_sparcity_list.append(update_sparcity)
            trace_sparcity_list.append(trace_sparcity)
        
        stack = lambda x: torch.stack(x, axis=0)
        spk_rec = stack(spk_rec_list)
        loss = stack(loss_list)
        sim_score = stack(sim_score_list)
        trace = stack(trace_list)
        update_sparcity = stack(update_sparcity_list)
        trace_sparcity = stack(trace_sparcity_list)

        output = TDCLOutput(
            loss=loss.mean().item(), 
            spikes_sum=spk_rec.reshape(T, B, -1).sum(dim=(0, 2)).mean().item(), 
            spikes_mean=spk_rec.mean().item(),
            sim_score_sum=sim_score.sum(dim=0).mean().item(),
            trace_sum=trace.reshape(T, B, -1).sum(dim=(0, 2)).mean().item(),
            trace_mean=trace.mean().item(),
            update_sparcity=update_sparcity.mean().item(),
            trace_sparcity=trace_sparcity.mean().item(),
        )

        return spk_rec, loss.mean(), output

    def train_step(self, x:torch.Tensor, input_sum:torch.Tensor, train:bool) -> tuple[torch.Tensor, float]:
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
        self.tdcl_layer.train()
        x, loss, output = self.step(x, input_sum, train)

        self.train_metrics.accumulate(output)

        return x, loss.item()
    
    def val_step(self, x:torch.Tensor, input_sum:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.tdcl_layer.eval()
        x, loss, output = self.step(x, input_sum, train=False)
        self.val_metrics.accumulate(output)
        return x, loss.item()
    
    def test_step(self, x:torch.Tensor, input_sum:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.tdcl_layer.eval()
        x, loss, output = self.step(x, input_sum, train=False)
        self.test_metrics.accumulate(output)
        return x, loss.item()
    
    def compute_metrics(self) -> None:
        self.train_metrics.compute()
        self.val_metrics.compute()
        self.test_metrics.compute()

    def reset(self) -> None:
        self.tdcl_layer.reset()

    def to(self, device:str) -> None:
        self.tdcl_layer = self.tdcl_layer.to(device)


class FullOnlineTDCLTrainer:
    def __init__(
            self, 
            train_loader: Iterable, 
            val_loader: Iterable, 
            test_loader: Iterable,
            path:Union[str, None],
        ) -> None:
        self.layer_trainers: list[FullOnlineTDCLLayerTrainer] = []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.path = path
        self.device = "cpu"

    def __getitem__(self, idx: int) -> FullOnlineTDCLLayerTrainer:
        return self.layer_trainers[idx]

    def add_layer_trainer(self, layer_trainer: FullOnlineTDCLLayerTrainer) -> None:
        self.layer_trainers.append(layer_trainer)

    def compute_metrics(self) -> None:
        for layer_trainer in self.layer_trainers:
            layer_trainer.compute_metrics()

    @torch.no_grad()
    def train(self, epochs: int, layer_in_pbar: int = 0, save:bool=True, log_offset:int=0, layer_to_train:Union[int, None]=None, train_lower_layers_too:bool=False, train_tsne:bool=False) -> tuple[torch.Tensor, list[float]]:
        """
        controlls all training, from running each batch through the layers, updating the layers independently, tracking each layers' metrics, and creating tsne plots
        """
        assert len(self.layer_trainers) > 0
        print(f"start training for {epochs} epochs...")
        pbar = trange(epochs)
        for epoch in pbar:
            create_tsne = (self.path is not None) and ((epoch % 5) == 0 or (epoch+1) == epochs)
            if create_tsne:
                xs = [list() for _ in self.layer_trainers]
                ys = [list() for _ in self.layer_trainers]
            
            self.reset()
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                T = x.shape[0]
                B = x.shape[1]
                input_sum = x.detach()
                input_sum = input_sum.reshape(T, B, -1)
                input_sum = input_sum.mean(dim=-1)
                for j, layer_trainer in enumerate(self.layer_trainers):
                    # input_sum = leaky_integrate(input_sum)
                    train_flag = (layer_to_train is None) or (layer_to_train == j) or (layer_to_train >= j and train_lower_layers_too)
                    x = x.detach()
                    x, loss = layer_trainer.train_step(x, input_sum, train_flag) # (T, B, C', H', W')
                
                    if create_tsne and train_tsne:
                        xs[j].append(x.mean(dim=0).flatten(start_dim=1).detach().cpu().numpy())  
                        ys[j].append(y.detach().cpu().numpy())

                pbar_dict = self.layer_trainers[layer_in_pbar].train_metrics.get_mean_buffer_values()
                pbar.set_postfix(batch=f"{i+1}/{len(self.train_loader)}", **pbar_dict)

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
                    x, loss = layer_trainer.val_step(x, input_sum) # (T, B, C', H', W')

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
                    x, loss = layer_trainer.test_step(x, input_sum) # (T, B, C', H', W')
            
                    if create_tsne and not train_tsne:
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
                    fig.savefig(fig_path + f"/epoch_{epoch + 1 + log_offset}_layer_{j+1}_tsne.png")
        
        if save:
            self.save_training(self.path + "final_checkpoint.pkl")

    def reset(self) -> None:
        for layer_trainer in self.layer_trainers:
            layer_trainer.reset()
    
    def to(self, device:str) -> "FullOnlineTDCLTrainer":
        for layer_trainer in self.layer_trainers:
            layer_trainer.to(device)
        self.device = device
        return self

    def save_training(self, path: str) -> None:
        checkpoint = {}
        for i, layer_trainer in enumerate(self.layer_trainers):
            checkpoint[f"layer_{i+1}"] = {}
            checkpoint[f"layer_{i+1}"]["state_dict"] = layer_trainer.tdcl_layer.state_dict()
            checkpoint[f"layer_{i+1}"]["train_metrics"] = layer_trainer.train_metrics
            checkpoint[f"layer_{i+1}"]["val_metrics"] = layer_trainer.val_metrics
            checkpoint[f"layer_{i+1}"]["test_metrics"] = layer_trainer.test_metrics
        dill.dump(checkpoint, open(path, "wb"))

    @classmethod
    def load(cls, path: str) -> "FullOnlineTDCLTrainer":
        """
        DEPRICATED,
        TODO: FIX
        """
        return dill.load(open(path, "rb"))