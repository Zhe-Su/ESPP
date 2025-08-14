import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from typing import *
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

class LI(nn.Module):
    def __init__(self, beta:float, carry_state:bool) -> None:
            super().__init__()
            self.beta = beta 
            self.carry_state = carry_state
            if self.carry_state:
                self.state = torch.tensor((0,))
            else:
                self.state = None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        input:
            x: (T, ...)
            beta: float
        output:
            li: (T, ...) same shape as x
        """ 
        T = x.shape[0]
        if self.carry_state:
            buff = self.state
        else:
            buff = 0
        
        li = []
        for t in range(T):
            buff = self.beta * buff + x[t]
            li.append(buff)
        li = torch.stack(li)# * (1-beta)

        if self.carry_state:
            self.state = buff

        return li
    
    def step(self, x:torch.Tensor) -> torch.Tensor:
        """
        input:
            x: (...)
            beta: float
        output:
            li: (...) same shape as x
        """ 
        assert self.carry_state
        self.state = self.beta * self.state + x
        return self.state
    
    def to(self, device:str) -> "LI":
        self.state = self.state.to(device)
        return super().to(device)
    
    def reset(self) -> None:
        if self.carry_state:
            device = self.state.device
            self.state = torch.tensor((0,)).to(device)


class Tracer:
    def __init__(self, carry_state:bool):
        self.carry_state = carry_state
        self.traces:list[tuple[torch.Tensor, LI]] = []
    
    def add(self, gain:float, beta:float) -> None:
        li = LI(beta, self.carry_state)
        self.traces.append([torch.tensor((gain,)), li])
        self.traces = sorted(self.traces, key=lambda x: x[1].beta, reverse=True)


    @torch.no_grad()
    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        out = 0
        for (gain, li) in self.traces:
            out += gain * li(x)
        return out
    
    
    def step(self, x:torch.Tensor) -> None:
        out = 0
        for (gain, li) in self.traces:
            out += gain * li.step(x)
    
    def get(self) -> torch.Tensor:
        out = 0
        for (gain, li) in self.traces:
            out += gain * li.state
        return out
    
    def __str__(self) -> str:
        out = ""
        for (gain, li) in self.traces:
            out += f"({gain.item()}_{li.beta})_"
        if self.carry_state:
            out += "carry"
        else:
            out = out[:-1]
        return out

    def to(self, device) -> "Tracer":
        for i in range(len(self.traces)):
            self.traces[i][0] = self.traces[i][0].to(device)
            self.traces[i][1] = self.traces[i][1].to(device)
        return self
    
    def reset(self) -> None:
        for (gain, li) in self.traces:
            li.reset()

    def clone(self) -> "Tracer":
        new_tracer = Tracer(self.carry_state)
        for (gain, li) in self.traces:
            new_tracer.traces.append([gain.clone(), LI(li.beta, self.carry_state)])
        return new_tracer