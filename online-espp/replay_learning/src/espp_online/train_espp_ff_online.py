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
from src.espp_online import OnlineESPPLayer, OnlineESPPLayerTrainer, OnlineESPPTrainer
from src.espp_online import create_espp_ff_block
from typing import Union


def main() -> None:
    print("setup dataloader and metrics")

    # path = "./bin/bptt/mi_online_ff/imu/two_layer/input_thr_no_log_temp05_lr8e-2_20_epochs_50_few_shot_replay/"
    # path = "./bin/bptt/mi_online_ff/imu/two_layer/input_thr_no_log_learned_temp_lr8e-2_lr4e-3_300_epochs_3_few_shot_replay/"
    # path = "./bin/bptt/mi_online_ff/imu/random_neg_input_thr_no_log_learned_temp_lr8e-2_lr4e-3_300_epochs_3_few_shot_replay/"
    # path = "./bin/bptt/mi_online_ff/imu/binary_input_thr_no_log_beta05_gamma00_gamma02_temp1_lr8e-2_lr4e-3_100_epochs_3_few_shot_ripple_replay20/"
    path = "./bin/bptt/mi_online_ff/imu/binary_input_thr_no_log_beta05_gamma1.0_gamma1.5_mult_temp05_lr8e-2_lr4e-3_100_epochs_3_few_shot_ripple_replay20_balanced_sampling/"
    # path = "./bin/bptt/mi_online_ff/imu/binary_no_training/"
    # path = "./bin/bptt/mi_online_ff/imu/no_training/"
    fig_path = path + "plots/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    BATCH_SIZE = 50
    EPOCHS = 100
    # lr = 8e-2
    # lr = 4e-4
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # train_loader, test_loader = create_nmnist_dataloaders(BATCH_SIZE, num_workers=2, time_window=10000, reset_cache=False, drop_last=True)
    train_loader, test_loader = train_loader, test_loader = create_imu_dataloaders(batch_size=BATCH_SIZE, window_length=40, stride=40, num_workers=2, drop_last=True, binary=True, balanced_sampling=True)
    # train_loader, test_loader = create_shd_dataloaders(BATCH_SIZE, num_workers=2, bins=250, reset_cache=False)
    x, y = next(iter(train_loader))
    print(f"data shape: {x.shape}")

    multi_metric_builder = MultiMetricBuilder()
    multi_metric_builder.add_metric("loss", lambda o: o.loss)
    multi_metric_builder.add_metric("loss_pos", lambda o: o.loss_pos)
    multi_metric_builder.add_metric("loss_neg", lambda o: o.loss_neg)
    multi_metric_builder.add_metric("spike_sum", lambda o: o.spikes_sum)
    multi_metric_builder.add_metric("spike_mean", lambda o: o.spikes_mean)
    multi_metric_builder.add_metric("sim_score_pos_sum", lambda o: o.sim_score_pos_sum)
    multi_metric_builder.add_metric("sim_score_neg_sum", lambda o: o.sim_score_neg_sum)
    multi_metric_builder.add_metric("echo_mean", lambda o: o.echo_mean)
    multi_metric_builder.add_metric("update_sparcity", lambda o: o.update_sparcity)
    multi_metric_builder.add_metric("echo_sparcity", lambda o: o.echo_sparcity)
    multi_metric_builder.add_metric("beta", lambda o: o.beta)
    multi_metric_builder.add_metric("temp", lambda o: o.temp)

    espp_layer_func = partial(
        create_espp_ff_block,
        beta=0.5, 
        spike_grad=surrogate.atan(),
        init_hidden=True,
        c_pos=1.0,
        c_neg=-1.0,
        input_thr=0.0,
        rule="gated_mi",
        use_replay=True,
        apply_weight=False,
        learn_temp=False,
        stream_data=False,
        temp=0.5
    )

    print("setup model and trainers")
    espp_layer_trainer_1 = OnlineESPPLayerTrainer(
        # espp_layer=espp_layer_func(in_features=2*34*34, out_features=512),
        espp_layer=espp_layer_func(in_features=60, out_features=512, gamma=1.0),
        lr=8e-2,
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )
    espp_layer_trainer_2 = OnlineESPPLayerTrainer(
        espp_layer=espp_layer_func(in_features=512, out_features=512, gamma=1.5),
        lr=4e-3,
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

    espp_trainer = OnlineESPPTrainer(train_loader, [], test_loader, path)
    espp_trainer.add_layer_trainer(espp_layer_trainer_1)
    espp_trainer.add_layer_trainer(espp_layer_trainer_2)
    # espp_trainer.add_layer_trainer(espp_layer_trainer_3)
    # checkpoint: dict[str, dict[str, Any]] = dill.load(open("./bin/bptt/mi_online_ff/imu/cascaded_learning_tau/final_checkpoint.pkl", "rb"))
    # espp_layer_trainer_1.espp_layer.load_state_dict(checkpoint["layer_1"]["state_dict"]) # input: (T, B, 2, 34, 34)

    print("start training...")
    espp_trainer = espp_trainer.to(device)
    espp_trainer.train(EPOCHS, layer_in_pbar=0, layer_to_train=None, log_offset=EPOCHS*0, train_lower_layers_too=True, train_tsne=True)
    # espp_trainer.train(EPOCHS, layer_in_pbar=1, layer_to_train=1, log_offset=EPOCHS, train_lower_layers_too=True, train_tsne=True)


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
    espp_trainer = OnlineESPPTrainer(test_loader, [], test_loader, path)
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