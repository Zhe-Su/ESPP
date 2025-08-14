import torch
from torch import nn
import snntorch as snn
from snntorch import utils
from typing import Callable


class FF_Block(nn.Module):
    def __init__(
            self, 
            in_features:int, 
            out_features:int, 
            beta:float, 
            spike_grad:Callable,
            init_hidden:bool,
        ) -> None:
        super().__init__()
        
        self.ff = nn.Linear(in_features, out_features, bias=False)
        # k = 3/torch.sqrt(torch.tensor(in_channels))
        # conv.weight = nn.init.uniform_(conv.weight, -k, k)
        nn.init.kaiming_uniform_(self.ff.weight, mode='fan_in', nonlinearity='tanh')
        self.leaky = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=init_hidden, output=True)

    def get_in_dim(self) -> torch.Tensor:
        return self.ff.in_features
    
    def get_out_dim(self) -> torch.Tensor:
        return self.ff.out_features

    def get_weight(self) -> torch.Tensor:
        return self.ff.weight
    
    def set_weight(self, weight:torch.Tensor) -> None:
        if isinstance(weight, torch.Tensor):
            weight = nn.Parameter(weight)
        self.ff.weight = weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: presynaptic spikes: (T, B, N) or (T, B, C, H, W)
        output:
            spk_rec: (T, B, N')
            mem_rec: (T, B, N')
        """
        T = x.shape[0] 
        B = x.shape[1]
        x = x.reshape(T, B, -1) 
        spk_rec = []
        mem_rec = []
        x = torch.vmap(self.ff)(x)
        for t in range(T):  # x.size(0) = number of time steps
            spk_out, mem_out = self.leaky(x[t])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
        
        spk_rec = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)
        return spk_rec, mem_rec
    
    def step(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: presynaptic spikes: (B, N) or (B, C, H, W)
        output:
            spk_rec: (B, N')
            mem_rec: (B, N')
        """
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = self.ff(x)
        spk_out, mem_out = self.leaky(x)
        return spk_out, mem_out
    

class CNN_Block(nn.Module):
    def __init__(
            self, 
            in_channels:int, 
            out_channels:int, 
            kernel_size:int, 
            pool_size:int,    
            beta:float, 
            spike_grad:Callable,
            init_hidden:bool,
            leaky_for_pooling:bool
        ) -> None:
        super().__init__()
        self.leaky_for_pooling = leaky_for_pooling
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # k = 3/torch.sqrt(torch.tensor(in_channels))
        # conv.weight = nn.init.uniform_(conv.weight, -k, k)
        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='tanh')
        self.leaky = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=init_hidden, output=True, learn_beta=False)
        self.pooling = nn.MaxPool2d(pool_size, return_indices=True)

        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: presynaptic spikes: (T, B, C, H, W)
        output:
            spk_rec: (T, B, C', H', W')
            mem_rec: (T, B, C', H', W')
        """
        T = x.shape[0] 
        spk_rec = []
        mem_rec = []
        # x = torch.vmap(self.conv)(x)
        for t in range(T):  # x.size(0) = number of time steps
            xt = self.conv(x[t])
            if self.leaky_for_pooling:
                spk_out, mem_out = self.leaky(xt)
                spk_out, indices = self.pooling(spk_out)
                future_mem_shape = spk_out.shape
                mem_out = mem_out.reshape(-1)
                mem_out = mem_out[indices]
                mem_out = mem_out.reshape(future_mem_shape)
            else:
                x_pool, indices = self.pooling(xt)
                spk_out, mem_out = self.leaky(x_pool)
                # spk_out = x_pool
                # mem_out = x_pool
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
        
        spk_rec = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)
        return spk_rec, mem_rec
    

    def step(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: presynaptic spikes: (B, N) or (B, C, H, W)
        output:
            spk_rec: (B, N')
            mem_rec: (B, N')
        """
        x = self.conv(x)
        if self.leaky_for_pooling:
            spk_out, mem_out = self.leaky(x)
            spk_out, indices = self.pooling(spk_out)
            future_mem_shape = spk_out.shape
            mem_out = mem_out.reshape(-1)
            mem_out = mem_out[indices]
            mem_out = mem_out.reshape(future_mem_shape)
        else:
            x_pool, indices = self.pooling(x)
            spk_out, mem_out = self.leaky(x_pool)
        return spk_out, mem_out




class CNN_Block_Chipready(nn.Module):
    def __init__(
            self, 
            in_channels:int, 
            out_channels:int, 
            kernel_size:int, 
            stride:int,
            beta:float, 
            spike_grad:Callable,
            init_hidden:bool,
        ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # k = 3/torch.sqrt(torch.tensor(in_channels))
        # conv.weight = nn.init.uniform_(conv.weight, -k, k)
        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='tanh')
        self.leaky = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=init_hidden, output=True)

        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: presynaptic spikes: (T, B, C, H, W)
        output:
            spk_rec: (T, B, C', H', W')
            mem_rec: (T, B, C', H', W')
        """
        T = x.shape[0] 
        spk_rec = []
        mem_rec = []
        x = torch.vmap(self.conv)(x)
        for t in range(T):  # x.size(0) = number of time steps
            spk_out, mem_out = self.leaky(x[t])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
        
        spk_rec = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)
        return spk_rec, mem_rec
    

    def step(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: presynaptic spikes: (B, N) or (B, C, H, W)
        output:
            spk_rec: (B, N')
            mem_rec: (B, N')
        """
        x = self.conv(x)
        spk_out, mem_out = self.leaky(x)
        return spk_out, mem_out


class Dropin2d(nn.Module):
    def __init__(self, p:float) -> None:
        super().__init__()
        assert p >= 0 and p <= 1
        self.drop = nn.Dropout2d(p)

    def forward(self, spk:torch.Tensor) -> torch.Tensor:
        spk = -spk + 1
        spk = self.drop(spk)
        spk = -spk + 1
        return spk 