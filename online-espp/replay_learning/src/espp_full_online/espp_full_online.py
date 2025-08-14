import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from typing import Callable, Iterable, Union
from src.util.metric import MultiMetric, OutputBase
from src.util.replay_buffers import ReplayBuffer, GMMReplayBuffer, RippleReplayBuffer
from dataclasses import dataclass
from tqdm import tqdm
import dill
import os 
import numpy as np
from src.util.create_tsne import plot_tsne
from tqdm import trange
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

@torch.jit.script
def calc_M(surr_mem_out:torch.Tensor, in_trace:torch.Tensor) -> torch.Tensor:
    # M = surr_mem_out[:,:,None] * in_trace[:,None,:] # (B, N', N)
    M = torch.einsum("bk, bl -> bkl", surr_mem_out,  in_trace) # (B, N', N)
    return M

@torch.jit.script
def update(dL, M, echo_pos, echo_neg, N:int):
    # M = surr_mem_out[:,:,None] * in_trace[:,None,:] # (B, N', N)
    # M_trace_pos = self.m_trace # (B, N', N)
    # M_trace_neg = self.trace_replay_buffer(M_trace_pos)
    
    # calculate gradients
    # dW_pos = echo_pos[:,:,None] * M # + M_trace_pos * spikes[:,:,None] # (B, N', N)
    dW_pos = torch.einsum("bk, bkl -> bkl", echo_pos, M) # + M_trace_pos * spikes[:,:,None] # (B, N', N)
    # dW_neg = echo_neg[:,:,None] * M # + M_trace_neg * spikes[:,:,None] # (B, N', N)
    dW_neg = torch.einsum("bk, bkl -> bkl", echo_neg, M) # + M_trace_pos * spikes[:,:,None] # (B, N', N)
    # dW = -(loss >= 0)[:, None, None].float() * (dW_pos - dW_neg)
    dW = -dL[:, None, None].float() * (dW_pos - dW_neg)
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


class FullOnlineESPPLayer(nn.Module):
    def __init__(
            self, 
            layer: nn.Module, 
            lr: float,
            c_pos:float=1., 
            c_neg:float=-1., 
            input_thr:float=0.02,
            use_replay:bool=True, 
            update_with_M_trace:bool=False,
            # stream_data:bool=False, 
            # apply_weight:bool=False, 
            # learn_temp:bool=False, 
            gamma:float=0.1,
            temp:float=0.5,
            K:int=1,
        ) -> None:
        super().__init__()
        assert c_pos > 0
        assert c_neg < 0
        self.layer = layer
        self.c_pos = c_pos#.detach()
        self.c_neg = c_neg
        self.input_thr = input_thr
        self.use_replay = use_replay
        self.update_with_M_trace = update_with_M_trace
        self.gamma = gamma
        self.temp = temp
        self.k = K

        # self.lr = 8e-2
        self.lr = lr

        self.device = "cpu"

        self.beta = self.layer.leaky.beta
        period = int(torch.tensor(0.5).log().item() / self.beta.log().item())
        size = 5
        # self.spike_replay_buffer = ReplayBuffer(size=size, period=period)
        self.spike_replay_buffer = RippleReplayBuffer(size=20)
        # self.trace_replay_buffer = ReplayBuffer(size=size, period=period)
        self.trace_replay_buffer = RippleReplayBuffer(size=20)

        self.init_traces = True
        self.in_trace = None
        self.out_trace = None
        self.m_trace = None
        # self.in_trace = torch.zeros([1, self.layer.get_in_dim()])
        # self.out_trace = torch.zeros([1, self.layer.get_out_dim()])
        # self.m_trace = torch.zeros([1, self.layer.get_out_dim(), self.layer.get_in_dim()])

        self.optim = torch.optim.SGD(self.parameters(), lr=4-4)

    def forward(self, x: torch.Tensor, y: torch.Tensor, input_sum:torch.Tensor, train:bool):
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
        rand_idx = torch.randperm(n=B)

        spk_out, mem_out = self.layer.step(x) # (B, N)
        spikes = spk_out.reshape([B, -1]) # (B, N')
        N = spikes.shape[-1]
        if self.init_traces:
            self.in_trace = torch.zeros(B, sum(x.shape[1:]), device=self.device)
            self.out_trace = torch.zeros(B, N, device=self.device)
            self.m_trace = torch.zeros(B, N, sum(x.shape[1:]), device=self.device)
            self.init_traces = False
            train = False
        
        # get echos
        echo_pos = self.out_trace
        if self.use_replay:
            echo_neg = self.spike_replay_buffer(echo_pos).to(self.device)
        else:
            echo_neg = echo_pos[rand_idx, ...]

        # calc sim scores
        sim_score_pos = spikes * echo_pos # (B, N')
        sim_score_pos = sim_score_pos.mean(axis=(-1)) # (B)
        
        sim_score_neg = spikes * echo_neg # (B, N')
        sim_score_neg = sim_score_neg.mean(axis=(-1)) # (B)
        
        # calc loss
        loss, loss_pos, loss_neg, update_sparcity = self.calc_mi_loss(sim_score_pos, sim_score_neg, input_sum)
        surr_mem_out = self.surr(mem_out-1)# * mem_out
        self.in_trace = self.beta * self.in_trace + x.reshape(B, -1)
        M = calc_M(surr_mem_out, self.in_trace)
        if train and (self.spike_replay_buffer.training_ready() or not self.use_replay):
            # update in trace

            W = self.layer.get_weight()
            dL = (loss > 0) * (input_sum > self.input_thr)
            
            if self.update_with_M_trace:
                M_trace_pos = self.m_trace # (B, N', N)
                if self.use_replay:
                    M_trace_neg = self.trace_replay_buffer(M_trace_pos).to(self.device)
                else:
                    M_trace_neg = self.m_trace[rand_idx, ...]
                
                dW = update_with_trace(dL, M, spikes, echo_pos.detach(), echo_neg.detach(), M_trace_pos.detach(), M_trace_neg.detach(), N)
            else:
                dW = update(dL, M, echo_pos.detach(), echo_neg.detach(), N)
            
            W = W - self.lr * dW
            self.layer.set_weight(W)

            # # calculate positive and negative M matrix 
            
            # # calculate gradients
            # dW_pos = echo_pos[:,:,None] * M + M_trace_pos * spikes[:,:,None] # (B, N', N)
            # dW_neg = echo_neg[:,:,None] * M + M_trace_neg * spikes[:,:,None] # (B, N', N)
            # dW = -(loss == 0).float() * (dW_pos - dW_neg)
            # dW = dW.mean(axis=0) # (N', N)
        
            # # update layer weights
            # W = self.layer.get_weight()
            # W = W - self.lr * dW / N
            # self.layer.set_weight(W)

        # update traces
        loss = loss.mean()
        self.out_trace = self.beta * self.out_trace + spikes
        if self.update_with_M_trace:
            self.m_trace = self.beta * self.m_trace + M

        # some metrics
        spk_echo = self.out_trace
        echo_sparcity = (spk_echo == 0).float().mean()
        return spk_out, loss, loss_pos, loss_neg, sim_score_pos, sim_score_neg, spk_echo, update_sparcity, echo_sparcity


    def surr(self, mem:torch.Tensor) -> torch.Tensor:
        # return (1 / torch.pi) * (1 / (1 +  (torch.pi*mem)**2)) 
        return 1 / (1 +  (torch.pi*mem)**2) 
        

    def calc_mi_loss(self, sim_score_pos: torch.Tensor, sim_score_neg: torch.Tensor, input_sum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # loss = torch.log(1 + input_sum) - sim_score_pos + sim_score_neg
        loss = self.gamma * input_sum - sim_score_pos + sim_score_neg

        loss = torch.where(loss > 0, loss, 0) # (B)
        loss = torch.where(input_sum > self.input_thr, loss, 0) # (B)

        update_sparcity = (loss == 0).float().mean()
        loss = loss#.mean()
        loss_pos = sim_score_pos.mean()
        loss_neg = sim_score_neg.mean()

        return loss, loss_pos, loss_neg, update_sparcity
    
    def calc_espp_loss(self, sim_score_pos: torch.Tensor, sim_score_neg: torch.Tensor, input_sum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        loss_pos = self.c_pos * input_sum[self.k:,...] - sim_score_pos # (T, B)
        loss_neg = self.c_neg * input_sum[self.k:,...] + sim_score_neg # (T, B)
        
        # loss_pos = self.c_pos * input_sum - sim_score
        # loss_neg = self.c_neg * input_sum + sim_score

        # max
        loss_pos = torch.where(loss_pos > 0, loss_pos, 0) 
        loss_neg = torch.where(loss_neg > 0, loss_neg, 0) 
        
        # input threshold
        thr = self.input_thr
        loss_pos = torch.where(input_sum[self.k:,...] > thr, loss_pos, 0) # (T, B)
        loss_neg = torch.where(input_sum[self.k:,...] > thr, loss_neg, 0) # (T, B)
        
        # reduction
        update_sparcity = ((loss_pos + loss_neg) == 0).float().mean()
        loss_pos = loss_pos.sum(dim=0).mean() # ()
        loss_neg = loss_neg.sum(dim=0).mean() # ()
        loss = loss_pos + loss_neg # ()
        return loss, loss_pos, loss_neg, update_sparcity
    
    def reset(self) -> None:
        utils.reset(self)
        self.spike_replay_buffer.reset()
        self.trace_replay_buffer.reset()
        self.in_trace = None
        self.out_trace = None
        self.m_trace = None
        self.init_traces = True

    def to(self, device:str) -> None:
        super().to(device)
        self.beta = self.beta.to(device)
        # self.in_trace = self.in_trace.to(device)
        # self.out_trace = self.out_trace.to(device)
        # self.m_trace = self.m_trace.to(device)
        self.device = device
        return self


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
    

class FullOnlineESPPLayerTrainer:
    def __init__(
            self, 
            espp_layer: FullOnlineESPPLayer, 
            train_metrics: MultiMetric,
            val_metrics: MultiMetric,
            test_metrics: MultiMetric,
        ) -> None:
        self.espp_layer = espp_layer

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

        acc_pos = (self.espp_layer.c_pos*input_sum[self.espp_layer.k:,...] - sim_score_pos) <= 0
        acc_neg = (self.espp_layer.c_neg*input_sum[self.espp_layer.k:,...] + sim_score_neg) <= 0
        acc_pos = acc_pos.sum() / y_espp.sum()
        acc_neg = acc_neg.sum() / ~y_espp.sum()

        acc = (acc_pos + acc_neg) / 2
        return acc.item(), acc_pos.item(), acc_neg.item()

    def step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor, train:bool) -> tuple[torch.Tensor, torch.Tensor, ESPPOutput]:
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
        loss_pos_list = []
        loss_neg_list = []
        sim_score_pos_list = []
        sim_score_neg_list = []
        spk_echo_list = []
        update_sparcity_list = []
        echo_sparcity_list = []
        for t in range(T):
            spk_rec, loss, loss_pos, loss_neg, sim_score_pos, sim_score_neg, spk_echo, update_sparcity, echo_sparcity = self.espp_layer(x[t,...], y, input_sum[t,...], train)
            spk_rec_list.append(spk_rec)
            loss_list.append(loss)
            loss_pos_list.append(loss_pos)
            loss_neg_list.append(loss_neg)
            sim_score_pos_list.append(sim_score_pos)
            sim_score_neg_list.append(sim_score_neg)
            spk_echo_list.append(spk_echo)
            update_sparcity_list.append(update_sparcity)
            echo_sparcity_list.append(echo_sparcity)
        
        stack = lambda x: torch.stack(x, axis=0)
        spk_rec = stack(spk_rec_list)
        loss = stack(loss_list)
        loss_pos = stack(loss_pos_list)
        loss_neg = stack(loss_neg_list)
        sim_score_pos = stack(sim_score_pos_list)
        sim_score_neg = stack(sim_score_neg_list)
        spk_echo = stack(spk_echo_list)
        update_sparcity = stack(update_sparcity_list)
        echo_sparcity = stack(echo_sparcity_list)

        output = ESPPOutput(
            loss=loss.mean().item(), 
            loss_pos=loss_pos.mean().item(), 
            loss_neg=loss_neg.mean().item(), 
            spikes_sum=spk_rec.reshape(T, B, -1).sum(dim=(0, 2)).mean().item(), 
            spikes_mean=spk_rec.mean().item(),
            sim_score_pos_sum=sim_score_pos.sum(dim=0).mean().item(),
            sim_score_neg_sum=sim_score_neg.sum(dim=0).mean().item(),
            echo_sum=spk_echo.reshape(T, B, -1).sum(dim=(0, 2)).mean().item(),
            echo_mean=spk_echo.mean().item(),
            update_sparcity=update_sparcity.mean().item(),
            echo_sparcity=echo_sparcity.mean().item(),
        )

        return spk_rec, loss.mean(), output

    def train_step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor, train:bool) -> tuple[torch.Tensor, float]:
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
        x, loss, output = self.step(x, y, input_sum, train)

        self.train_metrics.accumulate(output)

        return x, loss.item()
    
    def val_step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        x, loss, output = self.step(x, y, input_sum, train=False)
        self.val_metrics.accumulate(output)
        return x, loss.item()
    
    def test_step(self, x:torch.Tensor, y:torch.Tensor, input_sum:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        x, loss, output = self.step(x, y, input_sum, train=False)
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


class FullOnlineESPPTrainer:
    def __init__(
            self, 
            train_loader: Iterable, 
            val_loader: Iterable, 
            test_loader: Iterable,
            path:Union[str, None],
        ) -> None:
        self.layer_trainers: list[FullOnlineESPPLayerTrainer] = []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.path = path
        self.device = "cpu"

    def __getitem__(self, idx: int) -> FullOnlineESPPLayerTrainer:
        return self.layer_trainers[idx]

    def add_layer_trainer(self, layer_trainer: FullOnlineESPPLayerTrainer) -> None:
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
                    x, loss = layer_trainer.train_step(x, y, input_sum, train_flag) # (T, B, C', H', W')
                
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
    
    def to(self, device:str) -> "FullOnlineESPPTrainer":
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
    def load(cls, path: str) -> "FullOnlineESPPTrainer":
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