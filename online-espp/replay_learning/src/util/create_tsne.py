import dill
import torch
from torch import nn
import numpy as np
from src.util.dataloading import create_nmnist_dataloaders
from snntorch import utils
from src.util.metric import MultiMetric
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Iterable

def create_tsne(model:nn.Module, data_loader:Iterable, num_batches:Union[int, None]=None) -> tuple[plt.Figure, plt.Axes]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)

    ys = []
    embeds = []
    print("running through model...")
    with torch.no_grad():
        model.eval()
        for i, (x, y) in tqdm(enumerate(data_loader)):
            if num_batches is not None and i >= num_batches:
                break
            x = x.to(device)
            embed = model.inference(x)

            ys.append(y.cpu())
            embeds.append(embed.cpu())

        embeds = [x.mean(dim=0) for x in embeds]
        embeds = torch.concat(embeds, dim=0)
        ys = torch.concat(ys)

        print("create tsne plots...")
        if embeds.sum() == 0:
            print("tensors to embed are empty.")
        else:
            embeds, ys = tsne_ready_torch_to_numpy(embeds, ys)
            fig, ax = plot_tsne(embeds, ys)
            print("done.")
            return fig, ax
    
def tsne_ready_torch_to_numpy(embeds: torch.Tensor, ys: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    embeds = embeds.flatten(start_dim=1).numpy()
    ys = ys.numpy()
    return embeds, ys

def plot_tsne(embeds: np.ndarray, ys: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    tsne = TSNE(n_components=2, perplexity=30, init="random")
    tsne_embed = tsne.fit_transform(embeds)

    fig, ax = plt.subplots(1, 1)
    fig: plt.Figure
    ax: plt.Axes
    sns.scatterplot(
        x=tsne_embed[...,0], 
        y=tsne_embed[...,1], 
        hue=ys,
        palette=sns.color_palette("hls", ys.max() + 1),
        ax=ax
    )        
    ax.set_title("N-MNIST data T-SNE projection")
    return fig, ax
    

if __name__ == "__main__":
    path = "./bin/bptt/infonce/"
    model: nn.Module = dill.load(open(path + "net.pkl", "rb"))
    _, test_loader = create_nmnist_dataloaders(batch_size=50, num_workers=2, time_window=10000)
    fig, ax = create_tsne(path=path, data_loader=test_loader, num_batches=None)
    fig.savefig(path=path + "tsne.png")