import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Iterable, Callable
from tqdm import tqdm
from snntorch import utils

def create_logits(x: torch.Tensor, layers:Iterable[nn.Module], classifier:nn.Module) -> torch.Tensor:
    with torch.no_grad():
        for layer in layers:
            utils.reset(layer)  # resets hidden states for all LIF neurons in net
            x, _ = layer(x)
    x = x.flatten(start_dim=2) # (T, B, C*H*W)
    x = torch.vmap(classifier)(x) # (T, B, Cls)
    logits = x.mean(axis=0) # (B, Cls)
    return logits


def train_epoch(layers:Iterable[nn.Module], classifier: nn.Module, optimizer:torch.optim.Optimizer, loss_fn: Callable, train_loader: DataLoader, device:str) -> tuple[float, float]:
    losses = []
    accs = []
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        logits = create_logits(x, layers, classifier)
        loss = loss_fn(logits, y) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=-1) == y).float().mean()

        losses.append(loss.item())
        accs.append(acc.item())
    
    mean = lambda l: sum(l) / len(l)
    loss = mean(losses)
    acc = mean(accs)
    return loss, acc


def test_epoch(layers:Iterable[nn.Module], classifier: nn.Module, loss_fn: Callable, test_loader: DataLoader, device:str) -> tuple[float, float]:
    losses = []
    accs = []
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        logits = create_logits(x, layers, classifier)
        loss = loss_fn(logits, y) 

        acc = (logits.argmax(dim=-1) == y).float().mean()
        
        losses.append(loss.item())
        accs.append(acc.item())
    
    mean = lambda l: sum(l) / len(l)
    loss = mean(losses)
    acc = mean(accs)
    return loss, acc


def train_few_shot(layers:Iterable[nn.Module], features:int, num_classes:int, epochs:int, train_loader:DataLoader, test_loader:DataLoader, device:str="cpu") -> tuple[list[float], list[float], list[float], list[float], nn.Module]:
    # classifier = nn.Sequential(
    #     nn.Linear(in_features=features, out_features=features),
    #     nn.ReLU(),
    #     nn.Linear(in_features=features, out_features=num_classes),
    # )
    classifier = nn.Linear(in_features=features, out_features=num_classes)
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_accs = []
    train_losses = []
    test_accs = []
    test_losses = []

    for epoch in tqdm(range(epochs)):
        print("training...")
        train_loss, train_acc = train_epoch(layers, classifier, optimizer, loss_fn, train_loader, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print("testing...")
        test_loss, test_acc = test_epoch(layers, classifier, loss_fn, test_loader, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print("train loss is: ", train_losses[-1])
        print("test loss is: ", test_losses[-1])
        print("train acc is: ", train_accs[-1])
        print("test acc is: ", test_accs[-1])
    return train_losses, test_losses, train_accs, test_accs, classifier