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
from src.util.dataloading import create_nmnist_dataloaders, create_dvs_dataloaders, create_tennis_dataloaders
from src.util.few_shot_training import train_few_shot
from src.util.blocks import CNN_Block, CNN_Block_Chipready
from src.espp_online import OnlineESPPLayer, OnlineESPPLayerTrainer, OnlineESPPTrainer
from src.espp_online import create_espp_block, create_espp_block_chipready
from typing import Union

def main() -> None:
    print("setup dataloader and metrics")

    # path = "./bin/bptt/mi_online/tennis/four_layer/cnn_max_bins64_augm_size128_input_gamma_no_log_temp05_no_weight_lr4e-3_20_4_epochs_and_lower_50_few_shot_replay/"
    # path = "./bin/bptt/mi_online/dvs_half/three_layer/cnn_max_beta05_bins100_filter10000_augm_size128_input_gamma_no_log_temp05_no_weight_lr4e-3_20_4_epochs_and_lower_50_few_shot/"
    path = "./bin/bptt/mi_online/dvs_half/four_layer/cnn_max_beta05_bins50_filter10000_augm_size128_input_gamma1.1-1.4_temp05_with_weight_lr4e-3_20_4_epochs_and_lower_50_few_shot_softplus/"
    # path = "./bin/bptt/mi_online/dvs_half/two_layer/testing_only_binary_input_beta05_akhf/"
    # path = "./bin/bptt/mi_online/dvs_half/four_layer/no_training_cnn_max_beta095-03_bins100_filter10000/" # TODO: rerun...
    fig_path = path + "plots/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    BATCH_SIZE = 25
    EPOCHS = 20
    # lr = 8e-2
    lr = 4e-3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, test_loader = create_dvs_dataloaders(BATCH_SIZE, num_workers=4, bins=50, reset_cache=False, drop_last=True, filter_time=10000)
    # train_loader, test_loader = create_tennis_dataloaders(BATCH_SIZE, num_bins=64, num_workers=4, drop_last=True)
    x, y = next(iter(train_loader))
    x, y = next(iter(test_loader))
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
        use_replay=False,
        apply_weight=True, 
    )
    # espp_layer_func = partial(
    #     create_espp_block_chipready,
    #     beta=0.5, 
    #     spike_grad=surrogate.atan(),
    #     init_hidden=True,
    #     c_pos=1.0,
    #     c_neg=-1.0,
    #     input_thr=0.03,
    #     rule="espp",
    #     use_replay=True,
    #     stream_data=False,
    # )

    print("setup model and trainers")
    espp_layer_trainer_1 = OnlineESPPLayerTrainer(
        espp_layer=espp_layer_func(in_channels=2, out_channels=16, kernel_size=5, beta=0.5, gamma=1.1), # in_channels=2, out_channels=8, kernel_size=6, stride=4
        lr=lr,
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )
    espp_layer_trainer_2 = OnlineESPPLayerTrainer(
        espp_layer=espp_layer_func(in_channels=16, out_channels=32, kernel_size=5, beta=0.5, gamma=1.2), # in_channels=8, out_channels=20, kernel_size=4, stride=1
        lr=lr,
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )
    espp_layer_trainer_3 = OnlineESPPLayerTrainer(
        espp_layer=espp_layer_func(in_channels=32, out_channels=64, kernel_size=5, beta=0.5, gamma=1.3), # in_channels=8, out_channels=20, kernel_size=4, stride=1
        lr=lr,
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )
    espp_layer_trainer_4 = OnlineESPPLayerTrainer(
        espp_layer=espp_layer_func(in_channels=64, out_channels=128, kernel_size=5, beta=0.5, gamma=1.4), # in_channels=8, out_channels=20, kernel_size=4, stride=1
        lr=lr,
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )
    # beta = 0.95
    # spike_grad = surrogate.atan()
    # class Net(nn.Module):
    #     def __init__(self) -> None:
    #         super().__init__()
    #         self.net = nn.Sequential(
    #             nn.Conv2d(2, 16, 5),
    #             nn.MaxPool2d(2),
    #             snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    #             nn.Conv2d(16, 32, 5),
    #             nn.MaxPool2d(2),
    #             snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    #             nn.Conv2d(32, 64, 5),
    #             nn.MaxPool2d(2),
    #             snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    #             nn.Conv2d(64, 128, 5),
    #             nn.MaxPool2d(2),
    #             snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    #             # nn.Flatten(),
    #             # nn.Linear(128*4*4, 11),
    #             # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
    #         )
    #         self.conv = nn.Conv2d(64, 128, 5)
    #         self.leaky = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
    #     def forward(self, x:torch.Tensor) -> tuple[torch.Tensor]:
    #         T = x.shape[0]
    #         spks, mems = [], []
    #         for t in range(T):
    #             spk, mem = self.net(x[t])
    #             spks.append(spk)
    #             mems.append(mem)
    #         spks = torch.stack(spks)
    #         mems = torch.stack(mems)
    #         return spks, mems
            

    # net = Net().to(device)
    # layer = OnlineESPPLayer(
    #         layer=net,
    #         c_pos=1.0,
    #         c_neg=-1.0,
    #         input_thr=0.0,
    #         # rule=rule,
    #         use_replay=False,
    #         stream_data=False
    #     )

    # espp_layer_trainer_1 = OnlineESPPLayerTrainer(
    #     espp_layer=layer, # in_channels=2, out_channels=8, kernel_size=6, stride=4
    #     lr=lr,
    #     train_metrics=multi_metric_builder.set_tag("train").build(),
    #     val_metrics=multi_metric_builder.set_tag("val").build(),
    #     test_metrics=multi_metric_builder.set_tag("test").build(),
    # )
    

    espp_trainer = OnlineESPPTrainer(train_loader, [], test_loader, path)
    espp_trainer.add_layer_trainer(espp_layer_trainer_1)
    espp_trainer.add_layer_trainer(espp_layer_trainer_2)
    espp_trainer.add_layer_trainer(espp_layer_trainer_3)
    espp_trainer.add_layer_trainer(espp_layer_trainer_4)

    # print("load weights...")
    # espp_trainer.load(dill.load(open(path + "final_checkpoint.pkl", "rb")))

    print("start training...")
    espp_trainer = espp_trainer.to(device)
    espp_trainer.train(EPOCHS, layer_in_pbar=0, log_offset=EPOCHS*0, layer_to_train=0, train_lower_layers_too=True, train_tsne=True)
    espp_trainer.train(EPOCHS, layer_in_pbar=1, log_offset=EPOCHS*1, layer_to_train=1, train_lower_layers_too=True, train_tsne=True)
    espp_trainer.train(EPOCHS, layer_in_pbar=2, log_offset=EPOCHS*2, layer_to_train=2, train_lower_layers_too=True, train_tsne=True)
    espp_trainer.train(EPOCHS, layer_in_pbar=3, log_offset=EPOCHS*3, layer_to_train=3, train_lower_layers_too=True, train_tsne=True)


    print("Few Shot Training...")
    layers = [espp_layer_trainer_1.espp_layer.layer, espp_layer_trainer_2.espp_layer.layer, espp_layer_trainer_3.espp_layer.layer, espp_layer_trainer_4.espp_layer.layer]
    train_losses, test_losses, train_accs, test_accs, classifier = train_few_shot(layers, 2048, 11, 50, train_loader, test_loader, device) # 20*6*6
    checkpoint: dict[str, dict[str, Any]] = dill.load(open(path + "final_checkpoint.pkl", "rb"))
    key = "few_shot"
    checkpoint[key] = {}
    checkpoint[key]["train_loss"] = train_losses
    checkpoint[key]["test_loss"] = test_losses
    checkpoint[key]["train_acc"] = train_accs
    checkpoint[key]["test_acc"] = test_accs
    dill.dump(checkpoint, open(path + "final_checkpoint.pkl", "wb"))

    print("done.")


if __name__ == "__main__":
    main()