import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import numpy as np

from tqdm import tqdm, trange
import dill
from typing import Union, Callable, Any
import os
from src.infonce import InfoNCELayer
from src.util.dataloading import create_nmnist_dataloaders
from functools import partial


class Block(nn.Module):
    def __init__(
            self, 
            in_channels:int, 
            out_channels:int, 
            kernel_size:int, 
            pool_size:int,    
            beta:float, 
            spike_grad:Callable,
            init_hidden:bool,
            output_mem:bool,
            leaky_for_pooling:bool
        ) -> None:
        super().__init__()
        if output_mem and leaky_for_pooling:
            raise NotImplementedError("If LIF should output its membrane potential, it has to come after pooling.")
        self.output_mem = output_mem
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        leaky = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=init_hidden, output=output_mem)
        pooling = nn.MaxPool2d(pool_size)
        if leaky_for_pooling:
            self.layers = nn.Sequential(conv, leaky, pooling)
        else:
            self.layers = nn.Sequential(conv, pooling, leaky)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        input:
            x: presynaptic spikes: (T, B, C, H, W)
        output:
            spk_rec: (T, B, C', H', W')
            mem_rec: (T, B, C', H', W') | None
        """
        T = x.shape[0] 
        spk_rec = []
        mem_rec = []
        utils.reset(self.layers)  # resets hidden states for all LIF neurons in net
        for step in range(T):  # x.size(0) = number of time steps
            if self.output_mem:
                spk_out, mem_out = self.layers(x[step])
                mem_rec.append(mem_out)
            else:
                spk_out = self.layers(x[step])
            spk_rec.append(spk_out)
        
        spk_rec = torch.stack(spk_rec)
        if self.output_mem:
            mem_rec = torch.stack(mem_rec)
        else:
            mem_rec = None
        
        return spk_rec, mem_rec

def create_info_nce_layer(
        in_channels:int, 
        out_channels:int, 
        kernel_size:int, 
        pool_size:int,    
        beta:float, 
        spike_grad:Callable,
        init_hidden:bool,
        output_mem:bool,
        leaky_for_pooling:bool,
        leaky_enc:bool,
        spike_context:bool,
        apply_weight:bool,
    ) -> InfoNCELayer:
        block = Block(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            pool_size=pool_size,    
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=init_hidden,
            output_mem=output_mem,
            leaky_for_pooling=leaky_for_pooling
        )
        return InfoNCELayer(
            layer=block,
            in_channels=out_channels,
            out_channels=out_channels, 
            beta=beta, 
            spike_grad=spike_grad, 
            k=4, 
            leaky_enc=leaky_enc, 
            spike_context=spike_context,
            apply_weight=apply_weight,
        )

path = "./bin/bptt/infonce/per_layer_training/single_layer/membrane_context_with_weight/"

info_nce_layer_func = partial(
        create_info_nce_layer,
        kernel_size=5, 
        pool_size=2,    
        beta=0.5, 
        spike_grad=surrogate.atan(),
        init_hidden=True,
        output_mem=False,
        leaky_for_pooling=False,
        leaky_enc=True,
        spike_context=False,
        apply_weight=True,
    )

info_nce_layer_1 = info_nce_layer_func(in_channels=2, out_channels=12)
info_nce_layer_2 = info_nce_layer_func(in_channels=12, out_channels=32)

checkpoint: dict[str, dict[str, Any]] = dill.load(open(path + "final_checkpoint.pkl", "rb"))
info_nce_layer_1.load_state_dict(checkpoint["layer_1"]["state_dict"]) # input: (T, B, 2, 34, 34)
info_nce_layer_2.load_state_dict(checkpoint["layer_2"]["state_dict"]) # outputs: [T, B, 32, 5, 5]

layers = [info_nce_layer_1, info_nce_layer_2]

classifier = nn.Linear(in_features=32*5*5, out_features=10)
optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

BATCH_SIZE = 50
train_loader, test_loader = train_loader, test_loader = create_nmnist_dataloaders(BATCH_SIZE, num_workers=2, time_window=10000, reset_cache=False)

train_accs = []
train_losses = []
test_accs = []
test_losses = []

print("training...")
for x, y in tqdm(train_loader):
    with torch.no_grad():
        for layer in layers:
            x, loss, log_score_pos, log_score_neg = layer(x)
    # x: [T, B, 32, 5, 5]
    x = x.mean(dim=0)
    x = x.flatten(start_dim=1) # (B, 32*5*5)

    logits = classifier(x)
    loss = loss_fn(logits, y) 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = (logits.argmax(dim=-1) == y).float().mean()

    train_losses.append(loss.item())
    train_accs.append(acc.item())


print("testing...")
for x, y in tqdm(test_loader):
    with torch.no_grad():
        for layer in layers:
            x, loss, log_score_pos, log_score_neg = layer(x)
        # x: [T, B, 32, 5, 5]
        x = x.mean(dim=0)
        x = x.flatten(start_dim=1) # (B, 32*5*5)

        logits = classifier(x)
        loss = loss_fn(logits, y) 

        acc = (logits.argmax(dim=-1) == y).float().mean()

        test_losses.append(loss.item())
        test_accs.append(acc.item())

mean = lambda l: sum(l) / len(l)
train_loss = mean(train_losses)
train_acc = mean(train_accs)

test_loss = mean(test_losses)
test_acc = mean(test_accs)

print(f"train loss is:{train_loss}")
print(f"test loss is:{test_loss}")
print(f"train loss is:{train_acc}")
print(f"test acc is:{test_acc}")