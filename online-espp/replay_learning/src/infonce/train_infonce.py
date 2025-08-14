import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate
from snntorch import utils

import os
from typing import Union, Callable
from functools import partial
from src.util.dataloading import create_nmnist_dataloaders
from src.util.metric import MultiMetricBuilder
from src.infonce import InfoNCELayer, InfoNCELayerTrainer, InfoNCETrainer 

class Block(nn.Module):
    def __init__(
            self, 
            in_channels:int, 
            out_channels:int, 
            kernel_size:int, 
            pool_size:int,    
            beta:float, 
            spike_grad:Callable,
            init_hidden:bool,
            output_mem:bool,
            leaky_for_pooling:bool
        ) -> None:
        super().__init__()
        if output_mem and leaky_for_pooling:
            raise NotImplementedError("If LIF should output its membrane potential, it has to come after pooling.")
        self.output_mem = output_mem
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        leaky = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=init_hidden, output=output_mem)
        pooling = nn.MaxPool2d(pool_size)
        if leaky_for_pooling:
            self.layers = nn.Sequential(conv, leaky, pooling)
        else:
            self.layers = nn.Sequential(conv, pooling, leaky)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        input:
            x: presynaptic spikes: (T, B, C, H, W)
        output:
            spk_rec: (T, B, C', H', W')
            mem_rec: (T, B, C', H', W') | None
        """
        T = x.shape[0] 
        spk_rec = []
        mem_rec = []
        utils.reset(self.layers)  # resets hidden states for all LIF neurons in net
        for step in range(T):  # x.size(0) = number of time steps
            if self.output_mem:
                spk_out, mem_out = self.layers(x[step])
                mem_rec.append(mem_out)
            else:
                spk_out = self.layers(x[step])
            spk_rec.append(spk_out)
        
        spk_rec = torch.stack(spk_rec)
        if self.output_mem:
            mem_rec = torch.stack(mem_rec)
        else:
            mem_rec = None
        
        return spk_rec, mem_rec

def create_info_nce_layer(
        in_channels:int, 
        out_channels:int, 
        kernel_size:int, 
        pool_size:int,    
        beta:float, 
        spike_grad:Callable,
        init_hidden:bool,
        output_mem:bool,
        leaky_for_pooling:bool,
        leaky_enc:bool,
        spike_context:bool,
        apply_weight:bool,
    ) -> InfoNCELayer:
        block = Block(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            pool_size=pool_size,    
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=init_hidden,
            output_mem=output_mem,
            leaky_for_pooling=leaky_for_pooling
        )
        return InfoNCELayer(
            layer=block,
            in_channels=out_channels,
            out_channels=out_channels, 
            beta=beta, 
            spike_grad=spike_grad, 
            k=4, 
            leaky_enc=leaky_enc, 
            spike_context=spike_context,
            apply_weight=apply_weight,
        )

def main() -> None:
    print("setup dataloader and metrics")

    path = "./bin/bptt/infonce/per_layer_training/single_layer/membrane_context_with_weight/"
    fig_path = path + "plots/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    EPOCHS = 20
    BATCH_SIZE = 50
    train_loader, test_loader = create_nmnist_dataloaders(BATCH_SIZE, num_workers=2, time_window=10000, reset_cache=False)
    x, y = next(iter(train_loader))
    print(f"data shape: {x.shape}")

    multi_metric_builder = MultiMetricBuilder()
    multi_metric_builder.add_metric("loss", lambda o: o.loss)
    multi_metric_builder.add_metric("spikes_sum", lambda o: o.spikes_sum)
    multi_metric_builder.add_metric("spikes_mean", lambda o: o.spikes_mean)
    multi_metric_builder.add_metric("log_score_pos", lambda o: o.log_score_pos)
    multi_metric_builder.add_metric("log_score_neg", lambda o: o.log_score_neg)

    # neuron and simulation parameters
    info_nce_layer_func = partial(
        create_info_nce_layer,
        kernel_size=5, 
        pool_size=2,    
        beta=0.5, 
        spike_grad=surrogate.atan(),
        init_hidden=True,
        output_mem=False,
        leaky_for_pooling=False,
        leaky_enc=True,
        spike_context=False,
        apply_weight=True,
    )

    #  Initialize Network
    print("setup model and trainers")
    info_nce_layer_trainer_1 = InfoNCELayerTrainer(
        info_nce_layer=info_nce_layer_func(in_channels=2, out_channels=12),
        lr=8e-2,
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build()
    )
    info_nce_layer_trainer_2 = InfoNCELayerTrainer(
        info_nce_layer=info_nce_layer_func(in_channels=12, out_channels=32),
        lr=8e-2,
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build()
    )

    info_nce_trainer = InfoNCETrainer(
        train_loader=train_loader,
        val_loader=[],
        test_loader=test_loader,
        path=path

    )

    info_nce_trainer.add_layer_trainer(info_nce_layer_trainer_1)
    info_nce_trainer.add_layer_trainer(info_nce_layer_trainer_2)
    info_nce_trainer.to(device)
    info_nce_trainer.train(EPOCHS, layer_in_pbar= 0)

    print("done.")

    # dill.dump(model, open(path + "net.pkl", "wb"))
    # dill.dump(train_metrics, open(path + "train_metrics.pkl", "wb"))
    # dill.dump(test_metrics, open(path + "test_metrics.pkl", "wb"))

    # fig, ax = create_tsne(model, test_loader, num_batches=None)
    # fig.savefig(path + "plots/tsne.png")


if __name__ == "__main__":
    main()