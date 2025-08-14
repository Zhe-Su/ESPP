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
from src.util.dataloading import create_nmnist_dataloaders, create_shd_dataloaders
from src.util.few_shot_training import train_few_shot
from src.util.blocks import FF_Block
from src.espp import ESPPLayer, ESPPLayerTrainer, ESPPTrainer
from typing import Union, Any

def create_espp_block(
        in_features:int,
        out_features:int,
        beta:float, 
        spike_grad:Callable,
        init_hidden:bool,
        c_pos:float=2.0,
        c_neg:float=-1.0,
        input_thr:float=0.02,
    ) -> ESPPLayer:
        block = FF_Block(
            in_features=in_features,
            out_features=out_features,
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=init_hidden,
        )
        return ESPPLayer(
            layer=block,
            c_pos=c_pos,
            c_neg=c_neg,
            input_thr=input_thr,
        )

def main() -> None:
    print("setup dataloader and metrics")

    path = "./bin/bptt/espp_ff_old/per_layer_training/two_layer/avg_echo_c2_100_epochs/"
    fig_path = path + "plots/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    BATCH_SIZE = 128
    EPOCHS = 100
    lr = 8e-2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # train_loader, test_loader = create_nmnist_dataloaders(BATCH_SIZE, num_workers=2, time_window=10000, reset_cache=False)
    train_loader, test_loader = create_shd_dataloaders(BATCH_SIZE, num_workers=2, bins=250, reset_cache=True)
    x, y = next(iter(train_loader))
    print(f"data shape: {x.shape}")

    multi_metric_builder = MultiMetricBuilder()
    multi_metric_builder.add_metric("loss", lambda o: o.loss)
    multi_metric_builder.add_metric("loss_pos", lambda o: o.loss_pos)
    multi_metric_builder.add_metric("loss_neg", lambda o: o.loss_neg)
    # multi_metric_builder.add_metric("acc", lambda o: o.acc)
    # multi_metric_builder.add_metric("acc_pos", lambda o: o.acc_pos)
    # multi_metric_builder.add_metric("acc_neg", lambda o: o.acc_neg)
    multi_metric_builder.add_metric("spike_sum", lambda o: o.spikes_sum)
    multi_metric_builder.add_metric("spike_mean", lambda o: o.spikes_mean)
    multi_metric_builder.add_metric("sim_score_sum", lambda o: o.sim_score_sum)
    multi_metric_builder.add_metric("sim_score_mean", lambda o: o.sim_score_mean)
    multi_metric_builder.add_metric("echo_sum", lambda o: o.echo_sum)
    multi_metric_builder.add_metric("echo_mean", lambda o: o.echo_mean)
    multi_metric_builder.add_metric("update_sparcity", lambda o: o.update_sparcity)


    espp_layer_func = partial(
        create_espp_block,
        beta=0.95, 
        spike_grad=surrogate.atan(),
        init_hidden=True,
        c_pos=2.0,
        c_neg=-1.0,
        input_thr=0.02,
    )

    print("setup model and trascripts/run_espp.shiners")
    espp_layer_trainer_1 = ESPPLayerTrainer(
        espp_layer=espp_layer_func(in_features=700, out_features=512),
        lr=lr,
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )
    espp_layer_trainer_2 = ESPPLayerTrainer(
        espp_layer=espp_layer_func(in_features=512, out_features=512),
        lr=lr,
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )

    espp_trainer = ESPPTrainer(train_loader, [], test_loader, path)
    espp_trainer.add_layer_trainer(espp_layer_trainer_1)
    espp_trainer.add_layer_trainer(espp_layer_trainer_2)

    espp_trainer = espp_trainer.to(device)
    espp_trainer.train(EPOCHS, layer_in_pbar=0)

    print("Few Shot Training...")
    layers = [espp_layer_trainer_1.espp_layer.layer, espp_layer_trainer_2.espp_layer.layer]
    train_loss, test_loss, train_acc, test_acc = train_few_shot(layers, 512, 20, 30, train_loader, test_loader, device)
    checkpoint: dict[str, dict[str, Any]] = dill.load(open(path + "final_checkpoint.pkl", "rb"))
    checkpoint["few_shot"] = {}
    checkpoint["few_shot"]["train_loss"] = train_loss
    checkpoint["few_shot"]["test_loss"] = test_loss
    checkpoint["few_shot"]["train_acc"] = train_acc
    checkpoint["few_shot"]["test_acc"] = test_acc
    dill.dump(checkpoint, open(path + "final_checkpoint.pkl", "wb"))

    print("done.")


if __name__ == "__main__":
    main()
