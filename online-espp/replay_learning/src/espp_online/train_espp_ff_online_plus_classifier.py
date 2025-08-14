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
from src.util.blocks import FF_Block
from src.util.dataloading import create_nmnist_dataloaders, create_shd_dataloaders
from src.util.few_shot_training import train_few_shot
from src.espp_online import OnlineESPPLayer, OnlineESPPLayerTrainer, OnlineESPPTrainer
from src.espp_online import create_espp_ff_block
from typing import Union, Any
from tqdm import tqdm


def main() -> None:
    print("setup dataloader and metrics")

    # path = "./output/bptt/espp_online_ff/shd_spike_trace_input_mean_contrastive_no_reg_k1_input_thr_100_epochs/"
    path = "./output/bptt/mi_online_ff/shd_3layers_15input_thr_250_bins_beta095_100_epochs/"

    BATCH_SIZE = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, test_loader = create_nmnist_dataloaders(BATCH_SIZE, num_workers=2, time_window=10000, reset_cache=False)
    # train_loader, test_loader = create_shd_dataloaders(BATCH_SIZE, num_workers=2, bins=250, reset_cache=False)
    x, y = next(iter(train_loader))
    print(f"data shape: {x.shape}")

    espp_layer_func = partial(
        create_espp_ff_block,
        beta=0.95, 
        spike_grad=surrogate.atan(),
        init_hidden=True,
        c_pos=1.0,
        c_neg=-1.0,
        input_thr=0.03,
        rule="gated_mi",
        use_replay=True,
        stream_data=False,
    )

    espp_layer_1 = espp_layer_func(in_features=700, out_features=512)
    espp_layer_2 = espp_layer_func(in_features=512, out_features=512)

    checkpoint: dict[str, dict[str, Any]] = dill.load(open(path + "final_checkpoint.pkl", "rb"))
    espp_layer_1.load_state_dict(checkpoint["layer_1"]["state_dict"]) # input: (T, B, 2, 34, 34)
    espp_layer_2.load_state_dict(checkpoint["layer_2"]["state_dict"]) # outputs: [T, B, 32, 5, 5]

    espp_layer_1 = espp_layer_1.to(device)
    espp_layer_2 = espp_layer_2.to(device)

    layers = [espp_layer_1.layer, espp_layer_2.layer]
    # layers = [espp_layer_1.layer]

    train_loss, test_loss, train_acc, test_acc  = train_few_shot(layers, 512, 20, 30, train_loader, test_loader, device)
    # checkpoint: dict[str, dict[str, Any]] = dill.load(open(path + "final_checkpoint.pkl", "rb"))
    # checkpoint["few_shot"] = {}
    # checkpoint["few_shot"]["train_loss"] = train_loss
    # checkpoint["few_shot"]["test_loss"] = test_loss
    # checkpoint["few_shot"]["train_acc"] = train_acc
    # checkpoint["few_shot"]["test_acc"] = test_acc
    # dill.dump(checkpoint, open(path + "final_checkpoint.pkl", "wb"))


if __name__ == "__main__":
    main()