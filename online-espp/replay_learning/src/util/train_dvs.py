import torch
from torch import nn
from torch.utils.data import DataLoader
from functools import partial
from snntorch import surrogate
from snntorch import utils
from tqdm import tqdm
from typing import Iterable, Callable
from copy import deepcopy
import snntorch as snn

from src.util.dataloading import create_nmnist_dataloaders, create_dvs_dataloaders
from src.util.blocks import CNN_Block


def create_cnn_block(in_channels:int, out_channels:int, kernel_size:int) -> CNN_Block:
    return CNN_Block(
        in_channels,
        out_channels,
        kernel_size,
        pool_size=2,
        beta=0.95,
        spike_grad=surrogate.atan(),
        init_hidden=False,
        leaky_for_pooling=False,
    )

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = CNN_Block(
            in_channels=2,
            out_channels=8,
            kernel_size=5,
            pool_size=2,
            beta=0.95,
            spike_grad=surrogate.atan(),
            init_hidden=True,
            leaky_for_pooling=False,
        )
        self.layer2 = CNN_Block(
            in_channels=8,
            out_channels=32,
            kernel_size=5,
            pool_size=2,
            beta=0.95,
            spike_grad=surrogate.atan(),
            init_hidden=True,
            leaky_for_pooling=False,
        )
        self.layer3 = CNN_Block(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            pool_size=2,
            beta=0.95,
            spike_grad=surrogate.atan(),
            init_hidden=True,
            leaky_for_pooling=False,
        )
        # self.layer4 = create_cnn_block(in_channels=64, out_channels=64, kernel_size=5)
        self.flatten = nn.Flatten()
        self.class_head = nn.Linear(in_features=64*4*4, out_features=11)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        input:
            x: (B, C, H, W)
        output:
            x: (B, Cls)
        """
        # utils.reset(self.layer3.leaky)
        x1, _ = self.layer1.step(x)
        x2, _ = self.layer2.step(x1)
        x3, _ = self.layer3.step(x2)
        # x, _ = self.layer4(x)
        x4 = self.flatten(x3) # (B, N)
        x5 = self.class_head(x4) # (B, Cls)
        return x5
    
    def step(self, layer:CNN_Block, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = layer.conv(x)
        x, _ = layer.pooling(x)
        spk, mem = layer.leaky(x)
        return spk, mem


def model_call(model:Net, x:torch.Tensor) -> torch.Tensor:
    utils.reset(model)
    spks = []
    for t in range(x.shape[0]):
        spk = model(x[t])
        spks.append(spk)
    out = torch.stack(spks, axis=0)
    out = out.sum(axis=0)
    return out
    

def train_epoch(model:Net, optimizer:torch.optim.Optimizer, loss_fn: Callable, train_loader: DataLoader, device:str) -> tuple[float, float]:
    model.train()
    losses = []
    accs = []
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        # utils.reset(model)
        logits = model_call(model, x)
        loss: torch.Tensor = loss_fn(logits, y) 

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        acc = (logits.argmax(dim=-1) == y).float().mean()

        losses.append(loss.detach().item())
        accs.append(acc.detach().item())
    
    mean = lambda l: sum(l) / len(l)
    loss = mean(losses)
    acc = mean(accs)
    return loss, acc


@torch.no_grad()
def test_epoch(model:Net, loss_fn: Callable, test_loader: DataLoader, device:str) -> tuple[float, float]:
    model.eval()
    losses = []
    accs = []
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        utils.reset(model)
        logits = model_call(model, x)
        loss = loss_fn(logits, y) 

        acc = (logits.argmax(dim=-1) == y).float().mean()
        
        losses.append(loss.item())
        accs.append(acc.item())
    
    mean = lambda l: sum(l) / len(l)
    loss = mean(losses)
    acc = mean(accs)
    return loss, acc
    

def main() -> None:
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    BATCH_SIZE = 50
    EPOCHS = 50
    
    train_loader, test_loader = create_dvs_dataloaders(BATCH_SIZE, num_workers=4, bins=30, reset_cache=False)
    x, y = next(iter(train_loader))
    print(x.shape)
    
    beta = 0.95
    spike_grad = surrogate.atan()
    net = nn.Sequential(
        nn.Conv2d(2, 16, 5),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 5),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 5),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 5),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        # nn.MaxPool2d(2),
        # nn.Flatten(),
        # nn.Linear(128*4*4, 11),
        # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
    ).to(device)
    # net = torch.jit.script(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()


    train_accs = []
    train_losses = []
    test_accs = []
    test_losses = []

    for epoch in tqdm(range(EPOCHS)):
        print("training...")
        train_loss, train_acc = train_epoch(net, optimizer, loss_fn, train_loader, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print("testing...")
        test_loss, test_acc = test_epoch(net, loss_fn, test_loader, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print("train loss is: ", train_losses[-1])
        print("test loss is: ", test_losses[-1])
        print("train acc is: ", train_accs[-1])
        print("test acc is: ", test_accs[-1])
        print("\n")


if __name__ == "__main__":
    main()