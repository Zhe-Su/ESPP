import torch
from typing import *
from sklearn.mixture import GaussianMixture
import numpy as np

class ReplayBuffer:
    def __init__(self, size:int, period:int, thr:float=0.0, random:bool=False) -> None:
        self.period = period
        self.size = size
        self.thr = thr
        self.random = random

        self.queue = []
        self.intermediate_buffer = None
        self.next = 0
        self.timer = 0
        
    def __call__(self, value:torch.Tensor) -> torch.Tensor:
        """
        Inhabits all of the replay buffer's functionality. The value is added to the buffer if needed and a new value is sampled from it.
        Should be called at every timestep.
        """
        value = value.detach().cpu()
        self._try_to_add_to_queue(value)
        output = self._sample()
        if isinstance(output, type(False)):
            output = torch.zeros_like(value, device=value.device)
        self._update_queue()
        assert len(self.queue) <= self.size 
        return output

    def _try_to_add_to_queue(self, value:torch.Tensor) -> None:
        if value.mean() < self.thr:
            return
        if len(self.queue) < self.size:
            self.queue.append(value)
        elif self.intermediate_buffer is None:
            self.intermediate_buffer = value

    def _sample(self) -> Union[torch.Tensor, bool]:
        if len(self.queue) == 0:
            return False
        if self.random:
            mean = torch.stack(self.queue).mean(axis=0) # N
            value = torch.normal(mean=mean, std=1)
        else:
            value = self.queue[self.next]
            self.next = (self.next + 1) % self.size
        return value

    def _update_queue(self) -> None:
        if self.timer == self.period and self.intermediate_buffer is not None:
            self.queue = self.queue[1:]
            self.queue.append(self.intermediate_buffer)
            self.intermediate_buffer = None
            self.timer = 0
        elif self.intermediate_buffer is None:
            self.timer = 0
        else:
            self.timer += 1

    def training_ready(self) -> bool:
        return len(self.queue) >= self.size

    def reset(self) -> None:
        self.queue = []
        self.intermediate_buffer = None
        self.next = 0
        self.timer = 0


class RippleReplayBuffer:
    def __init__(self, size:int, thr:float=0.0) -> None:
        self.size = size
        self.thr = thr
        self.queue = []
        
    def __call__(self, value:torch.Tensor) -> torch.Tensor:
        """
        Inhabits all of the replay buffer's functionality. The value is added to the buffer if needed and a new value is sampled from it.
        Should be called at every timestep.
        """
        value = value.detach().cpu()
        self._try_to_add_to_queue(value)
        output = self._sample()
        if isinstance(output, type(False)):
            output = torch.zeros_like(value, device=value.device)
        self._update_queue()
        assert len(self.queue) <= self.size 
        return output

    def _try_to_add_to_queue(self, value:torch.Tensor) -> None:
        if value.mean() < self.thr:
            return
        self.queue.append(value)
            

    def _sample(self) -> Union[torch.Tensor, bool]:
        if len(self.queue) < self.size:
            return False
        value = self.queue[0]
        # pos = self.queue[-1]
        # values = torch.stack(self.queue)
        # scores = (pos[None, ...] * values)
        # for _ in range(len(scores.shape) - 1):
        #     scores = scores.sum(-1)
        # index = torch.argmin(scores)
        # value = self.queue[index]
        return value

    def _update_queue(self) -> None:
        if len(self.queue) > self.size:
            self.queue = self.queue[1:]

    def training_ready(self) -> bool:
        return len(self.queue) >= self.size

    def reset(self) -> None:
        self.queue = []



class GMMReplayBuffer:
    def __init__(self, size:int, period:int, thr:float=0.0) -> None:
        self.period = period
        self.size = size
        self.thr = thr

        self.queue = []
        self.timer = 0
        self.gmm = GaussianMixture(n_components=size)
        self.trained = False

    def __call__(self, value:torch.Tensor) -> torch.Tensor:
        """
        Inhabits all of the replay buffer's functionality. The value is added to the buffer if needed and a new value is sampled from it.
        Should be called at every timestep.
        """
        value = value.detach().cpu()
        self._try_to_add_to_queue(value)
        self._learn_mixture()
        output = self._sample()
        if isinstance(output, type(False)):
            output = torch.zeros_like(value, device=value.device)
        return output

    def _try_to_add_to_queue(self, value:torch.Tensor) -> None:
        if value.mean() < self.thr:
            return
        self.queue.append(value)

    def _sample(self) -> Union[torch.Tensor, bool]:
        if not self.trained:
            return False
        value, _ = self.gmm.sample()
        value = torch.from_numpy(value)
        return value

    def _learn_mixture(self) -> None:
        if self.timer == self.period:
            X = torch.concat(self.queue)
            self.gmm.fit(X)

            self.timer = 0
            self.queue = []
            self.trained = True
        else:
            self.timer += 1

    def training_ready(self) -> bool:
        return self.trained

    def reset(self) -> None:
        self.gmm = GaussianMixture(n_components=self.size)
        self.queue = []
        self.trained = False
        self.timer = 0