import torch
from torch import nn
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

import dill
import os
from typing import Callable
from functools import partial

from src.util.metric import MultiMetricBuilder
from src.util.blocks import CNN_Block, CNN_Block_Chipready
from src.util.dataloading import create_nmnist_dataloaders
from src.util.few_shot_training import train_few_shot
from src.espp_online import OnlineESPPLayer, OnlineESPPLayerTrainer, OnlineESPPTrainer
from src.espp_online import create_espp_block, create_espp_block_chipready
from typing import Union, Any
from tqdm import tqdm


def main() -> None:
    print("setup dataloader and metrics")

    path = "./output/bptt/espp_online/per_layer_training/two_layer/spike_trace_input_mean_100_epochs/"

    BATCH_SIZE = 50
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, test_loader = create_nmnist_dataloaders(BATCH_SIZE, num_workers=2, time_window=10000, reset_cache=False)
    x, y = next(iter(train_loader))
    print(f"data shape: {x.shape}")

    espp_layer_func = partial(
        create_espp_block,
        kernel_size=5, 
        pool_size=2,    
        # beta=0.95, 
        spike_grad=surrogate.atan(),
        init_hidden=True,
        leaky_for_pooling=True,
        c_pos=1.0,
        c_neg=-1.0,
        input_thr=0.0,
        use_replay=True,
    )

    espp_layer_1 = espp_layer_func(in_channels=2, out_channels=12)
    espp_layer_2 = espp_layer_func(in_channels=12, out_channels=32)

    checkpoint: dict[str, dict[str, Any]] = dill.load(open(path + "final_checkpoint.pkl", "rb"))
    espp_layer_1.load_state_dict(checkpoint["layer_1"]["state_dict"]) # input: (T, B, 2, 34, 34)
    espp_layer_2.load_state_dict(checkpoint["layer_2"]["state_dict"]) # outputs: [T, B, 32, 5, 5]

    espp_layer_1 = espp_layer_1.to(device)
    espp_layer_2 = espp_layer_2.to(device)

    layers = [espp_layer_1.layer, espp_layer_2.layer]

    train_few_shot(layers, 32*5*5, 10, 1, train_loader, test_loader, device)


if __name__ == "__main__":
    main()