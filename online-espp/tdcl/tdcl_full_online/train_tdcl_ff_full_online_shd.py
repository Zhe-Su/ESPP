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

from tdcl.util.metric import MultiMetricBuilder
from tdcl.util.tracer import Tracer
from tdcl.util.dataloading import create_nmnist_dataloaders, create_shd_dataloaders, create_imu_dataloaders, create_fortiss_dataloaders
from tdcl.util.few_shot_training import train_few_shot, train_epoch, test_epoch
from tdcl.tdcl_full_online import FullOnlineTDCLLayer, FullOnlineTDCLLayerTrainer, FullOnlineTDCLTrainer
from typing import Union

class Block(nn.Module):
    def __init__(
            self, 
            in_features:int, 
            out_features:int, 
            beta:float, 
            spike_grad:Callable,
            init_hidden:bool,
        ) -> None:
        super().__init__()
        
        self.ff = nn.Linear(in_features, out_features, bias=False)
        nn.init.kaiming_uniform_(self.ff.weight, mode='fan_in', nonlinearity='tanh')
        # self.ff.weight = torch.nn.Parameter(self.ff.weight*3)
        self.leaky = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=init_hidden, output=True, learn_beta=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        input:
            x: presynaptic spikes: (T, B, C, H, W)
        output:
            spk_rec: (T, B, C', H', W')
            mem_rec: (T, B, C', H', W') | None
        """
        T = x.shape[0] 
        B = x.shape[1]
        x = x.reshape(T, B, -1) 
        spk_rec = []
        mem_rec = []
        utils.reset(self)  # resets hidden states for all LIF neurons in net
        for t in range(T):  # x.size(0) = number of time steps
            spk_out, mem_out = self.leaky(self.ff(x[t]))
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
        
        spk_rec = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)
        return spk_rec, mem_rec
    
    def step(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: presynaptic spikes: (B, N) or (B, C, H, W)
        output:
            spk_rec: (B, N')
            mem_rec: (B, N')
        """
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = self.ff(x)
        spk_out, mem_out = self.leaky(x)
        return spk_out, mem_out
    
    def get_in_dim(self) -> torch.Tensor:
        return self.ff.in_features
    
    def get_out_dim(self) -> torch.Tensor:
        return self.ff.out_features

    def get_weight(self) -> torch.Tensor:
        return self.ff.weight
    
    def set_weight(self, weight:torch.Tensor) -> None:
        if isinstance(weight, torch.Tensor):
            weight = nn.Parameter(weight)
        self.ff.weight = weight



def create_tdcl_block(
        in_features:int, 
        out_features:int, 
        beta:float, 
        spike_grad:Callable,
        init_hidden:bool,
        lr: float,
        tracer:Tracer,
        warm_up:int,
        input_thr:float=0.02,
        gamma:float=0.1,
        temp:float=0.5,
    ) -> FullOnlineTDCLLayer:
        block = Block(
            in_features=in_features, 
            out_features=out_features, 
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=init_hidden,
        )
        return FullOnlineTDCLLayer(
            layer=block, 
            tracer=tracer,
            lr=lr,
            warm_up=warm_up,
            input_thr=input_thr,
            gamma=gamma,
            temp=temp,
        )

def main() -> None:
    print("setup dataloader and metrics")
    tracer = Tracer(carry_state=True)
    tracer.add(gain=-0.8714, beta=0.9)
    tracer.add(gain=1.1816, beta=0.85)
    # tracer.add(gain=-5.8644, beta=0.4)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    BATCH_SIZE = 1
    EPOCHS = 100
    beta = 0.5
    bins = 100
    input_thr = 0.01
    warm_up = 20
    classifier_epochs = 3
    num_targets = 20

    path = "./bin/tdcl_full_online/shd/ff/B1_bins"+str(bins)+"_beta"+str(beta)+"_"+str(tracer)+"_warmp_up"+str(warm_up)+"_inp_thr"+str(input_thr)+"_lr(8e-2_4e-3)_tmp(0.5_0.25)_gamma1.0_epochs"+str(EPOCHS)+"/"
    if path[-1] != "/":
        path += "/" 
    fig_path = path + "plots/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    train_loader, test_loader = create_shd_dataloaders(batch_size=BATCH_SIZE, bins=bins, num_workers=4)
    x, y = next(iter(train_loader))
    print(f"data shape: {x.shape}")

    multi_metric_builder = MultiMetricBuilder()
    multi_metric_builder.add_metric("loss", lambda o: o.loss)
    multi_metric_builder.add_metric("spike_sum", lambda o: o.spikes_sum)
    multi_metric_builder.add_metric("spike_mean", lambda o: o.spikes_mean)
    multi_metric_builder.add_metric("sim_score_sum", lambda o: o.sim_score_sum)
    multi_metric_builder.add_metric("trace_mean", lambda o: o.trace_mean)
    multi_metric_builder.add_metric("update_sparcity", lambda o: o.update_sparcity)
    multi_metric_builder.add_metric("trace_sparcity", lambda o: o.trace_sparcity)

    tdcl_layer_func = partial(
        create_tdcl_block,
        beta=beta, 
        spike_grad=surrogate.atan(),
        init_hidden=True,
        input_thr=input_thr,
        warm_up=warm_up,
    )

    print("setup model and trainers")
    tdcl_layer_trainer_1 = FullOnlineTDCLLayerTrainer(
        tdcl_layer=tdcl_layer_func(in_features=700, out_features=512, lr=8e-2, gamma=1.0, temp=0.5, tracer=tracer.clone()),
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )
    tdcl_layer_trainer_2 = FullOnlineTDCLLayerTrainer(
        tdcl_layer=tdcl_layer_func(in_features=512, out_features=512, lr=4e-3, gamma=1.0, temp=0.25, tracer=tracer.clone()),
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )

    # save_path = "./bin/bptt/mi_full_online_ff/shd/B1_bins100_binary_input_thr5%_no_log_gamma1.0_gamma1.0_temp05_temp025_lr8e-2_lr4e-3_10_epochs_3_few_shot_ripple_replay20/"
    # checkpoint: dict[str, dict[str, Any]] = dill.load(open(path + "final_checkpoint.pkl", "rb"))
    # tdcl_layer_trainer_1.tdcl_layer.load_state_dict(checkpoint["layer_1"]["state_dict"])
    # tdcl_layer_trainer_2.tdcl_layer.load_state_dict(checkpoint["layer_2"]["state_dict"])

    tdcl_trainer = FullOnlineTDCLTrainer(train_loader, [], test_loader, path)
    tdcl_trainer.add_layer_trainer(tdcl_layer_trainer_1)
    tdcl_trainer.add_layer_trainer(tdcl_layer_trainer_2)
    # tdcl_trainer.add_layer_trainer(tdcl_layer_trainer_3)

    tdcl_trainer = tdcl_trainer.to(device)
    tdcl_trainer.train(EPOCHS, layer_in_pbar=0, layer_to_train=None, log_offset=EPOCHS*0, train_lower_layers_too=True, train_tsne=False)

    print("Few Shot Training...")
    layers = [tdcl_layer_trainer_1.tdcl_layer.layer, tdcl_layer_trainer_2.tdcl_layer.layer]
    train_loss, test_loss, train_acc, test_acc, classifier = train_few_shot(layers, 512, num_targets, classifier_epochs, train_loader, test_loader, device)
    checkpoint: dict[str, dict[str, Any]] = dill.load(open(path + "final_checkpoint.pkl", "rb"))
    checkpoint["few_shot"] = {}
    checkpoint["few_shot"]["train_loss"] = train_loss
    checkpoint["few_shot"]["test_loss"] = test_loss
    checkpoint["few_shot"]["train_acc"] = train_acc
    checkpoint["few_shot"]["test_acc"] = test_acc
    dill.dump(checkpoint, open(path + "final_checkpoint.pkl", "wb"))


    print("\nSimulate Online Learning")
    tdcl_trainer = FullOnlineTDCLTrainer(test_loader, [], test_loader, path)
    tdcl_trainer.add_layer_trainer(tdcl_layer_trainer_1)
    tdcl_trainer.add_layer_trainer(tdcl_layer_trainer_2)
    # tdcl_trainer.add_layer_trainer(tdcl_layer_trainer_3)

    tdcl_trainer = tdcl_trainer.to(device)
    tdcl_trainer.train(1, layer_in_pbar=0, save=False, log_offset=EPOCHS+1)
    train_loss, test_loss, train_acc, test_acc, classifier = train_few_shot(layers, 512, num_targets, classifier_epochs, train_loader, test_loader, device)
    print("Test loss after training: ", test_loss)
    print("Test acc after training: ", test_acc)
    checkpoint["few_shot_online_sim"] = {}
    checkpoint["few_shot_online_sim"]["train_loss"] = train_loss
    checkpoint["few_shot_online_sim"]["test_loss"] = test_loss
    checkpoint["few_shot_online_sim"]["train_acc"] = train_acc
    checkpoint["few_shot_online_sim"]["test_acc"] = test_acc
    dill.dump(checkpoint, open(path + "final_checkpoint.pkl", "wb"))
    
    print("done.")


if __name__ == "__main__":
    main()