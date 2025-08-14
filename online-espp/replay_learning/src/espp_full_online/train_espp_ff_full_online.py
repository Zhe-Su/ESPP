import torch
from torch import nn
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

import dill
import os
from typing import Callable, Any
from functools import partial

from src.util.metric import MultiMetricBuilder
from src.util.blocks import FF_Block
from src.util.dataloading import create_nmnist_dataloaders, create_shd_dataloaders, create_imu_dataloaders
from src.util.few_shot_training import train_few_shot, train_epoch, test_epoch
from src.espp_full_online import FullOnlineESPPLayer, FullOnlineESPPLayerTrainer, FullOnlineESPPTrainer
from typing import Union


def create_espp_block(
        in_features:int, 
        out_features:int, 
        beta:float, 
        spike_grad:Callable,
        init_hidden:bool,
        lr: float,
        c_pos:float=1., 
        c_neg:float=-1., 
        input_thr:float=0.02,
        use_replay:bool=True, 
        update_with_M_trace:bool=False,
        # stream_data:bool=False, 
        # apply_weight:bool=False, 
        # learn_temp:bool=False, 
        gamma:float=0.1,
        temp:float=0.5,
        K:int=1,
    ) -> FullOnlineESPPLayer:
        block = FF_Block(
            in_features=in_features, 
            out_features=out_features, 
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=init_hidden,
        )
        return FullOnlineESPPLayer(
            layer=block,
            lr=lr,
            c_pos=c_pos,
            c_neg=c_neg,
            input_thr=input_thr,
            use_replay=use_replay,
            update_with_M_trace=update_with_M_trace,
            gamma=gamma,
            temp=temp,
            K=K,
        )

def main() -> None:
    print("setup dataloader and metrics")

    # path = "./bin/bptt/mi_full_online_ff/per_layer_training/two_layer/spike_trace_input_mean_contrastive_no_reg_k1_input_thr_replay_stream_20_epochs_full/"
    path = "./bin/bptt/mi_full_online_ff/imu/B1_binary_input_thr_no_log_gamma1.0_gamma1.0_temp05_temp025_lr8e-2_lr4e-3_100_epochs_3_few_shot_ripple_replay20_retrained_last20_layer_input_sum/"
    fig_path = path + "plots/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    BATCH_SIZE = 1
    EPOCHS = 20
    lr = 1e-3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

    # train_loader, test_loader = create_nmnist_dataloaders(BATCH_SIZE, num_workers=2, time_window=10000, reset_cache=False, drop_last=True)
    train_loader, test_loader = create_imu_dataloaders(batch_size=BATCH_SIZE, window_length=40, stride=40, num_workers=2, drop_last=True, binary=True, balanced_sampling=False)
    # train_loader, test_loader = create_shd_dataloaders(BATCH_SIZE, num_workers=2, bins=250, reset_cache=False)
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
    multi_metric_builder.add_metric("sim_score_pos_sum", lambda o: o.sim_score_pos_sum)
    multi_metric_builder.add_metric("sim_score_neg_sum", lambda o: o.sim_score_neg_sum)
    # multi_metric_builder.add_metric("echo_sum", lambda o: o.echo_sum)
    multi_metric_builder.add_metric("echo_mean", lambda o: o.echo_mean)
    multi_metric_builder.add_metric("update_sparcity", lambda o: o.update_sparcity)
    multi_metric_builder.add_metric("echo_sparcity", lambda o: o.echo_sparcity)

    espp_layer_func = partial(
        create_espp_block,
        beta=0.5, 
        spike_grad=surrogate.atan(),
        init_hidden=True,
        c_pos=1.0,
        c_neg=-1.0,
        input_thr=0.00,
        use_replay=True, 
        update_with_M_trace=False,
        # stream_data:bool=False, update_sparcity
        # apply_weight:bool=False, 
        # learn_temp:bool=False, 
        temp=0.5,
        K=1,
    )

    print("setup model and trainers")
    espp_layer_trainer_1 = FullOnlineESPPLayerTrainer(
        # espp_layer=espp_layer_func(in_features=2*34*34, out_features=512, lr=4e-3),
        espp_layer=espp_layer_func(in_features=60, out_features=512, lr=8e-2, gamma=1.0),
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )
    espp_layer_trainer_2 = FullOnlineESPPLayerTrainer(
        espp_layer=espp_layer_func(in_features=512, out_features=512, lr=4e-3, gamma=1.0, temp=0.25),
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )
    # espp_layer_trainer_3 = OnlineESPPLayerTrainer(
    #     espp_layer=espp_layer_func(in_features=512, out_features=512),
    #     lr=lr,
    #     train_metrics=multi_metric_builder.set_tag("train").build(),
    #     val_metrics=multi_metric_builder.set_tag("val").build(),
    #     test_metrics=multi_metric_builder.set_tag("test").build(),
    # )
    save_path = "./bin/bptt/mi_full_online_ff/imu/B1_binary_input_thr_no_log_gamma00_gamma02_temp1_lr8e-2_lr4e-3_100_epochs_3_few_shot_ripple_replay20/"
    checkpoint: dict[str, dict[str, Any]] = dill.load(open(save_path + "final_checkpoint.pkl", "rb"))
    espp_layer_trainer_1.espp_layer.load_state_dict(checkpoint["layer_1"]["state_dict"])

    espp_trainer = FullOnlineESPPTrainer(train_loader, [], test_loader, path)
    espp_trainer.add_layer_trainer(espp_layer_trainer_1)
    espp_trainer.add_layer_trainer(espp_layer_trainer_2)
    # espp_trainer.add_layer_trainer(espp_layer_trainer_3)

    espp_trainer = espp_trainer.to(device)
    espp_trainer.train(EPOCHS, layer_in_pbar=1, layer_to_train=1, log_offset=EPOCHS*0, train_lower_layers_too=True, train_tsne=True)

    print("Few Shot Training...")
    layers = [espp_layer_trainer_1.espp_layer.layer, espp_layer_trainer_2.espp_layer.layer, espp_layer_trainer_2.espp_layer.layer]
    train_loss, test_loss, train_acc, test_acc, classifier = train_few_shot(layers, 512, 6, 3, train_loader, test_loader, device)
    checkpoint: dict[str, dict[str, Any]] = dill.load(open(path + "final_checkpoint.pkl", "rb"))
    checkpoint["few_shot"] = {}
    checkpoint["few_shot"]["train_loss"] = train_loss
    checkpoint["few_shot"]["test_loss"] = test_loss
    checkpoint["few_shot"]["train_acc"] = train_acc
    checkpoint["few_shot"]["test_acc"] = test_acc
    dill.dump(checkpoint, open(path + "final_checkpoint.pkl", "wb"))


    print("\nSimulate Online Learning")
    espp_trainer = FullOnlineESPPTrainer(test_loader, [], test_loader, path)
    espp_trainer.add_layer_trainer(espp_layer_trainer_1)
    espp_trainer.add_layer_trainer(espp_layer_trainer_2)
    # espp_trainer.add_layer_trainer(espp_layer_trainer_3)

    espp_trainer = espp_trainer.to(device)
    espp_trainer.train(1, layer_in_pbar=0, save=False, log_offset=EPOCHS+1)
    test_loss, test_acc = test_epoch(layers, classifier, nn.CrossEntropyLoss(), test_loader, device)
    print("Test loss after training: ", test_loss)
    print("Test acc after training: ", test_acc)
    checkpoint["few_shot_trained"] = {}
    checkpoint["few_shot_trained"]["test_loss"] = test_loss
    checkpoint["few_shot_trained"]["test_acc"] = test_acc
    dill.dump(checkpoint, open(path + "final_checkpoint.pkl", "wb"))
    
    print("done.")


if __name__ == "__main__":
    main()