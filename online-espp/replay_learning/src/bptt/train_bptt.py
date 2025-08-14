import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

import tqdm
from dataclasses import dataclass
import dill
import os

from src.util.dataloading import create_nmnist_dataloaders
from src.util.metric import MultiMetricBuilder, MultiMetric, OutputBase


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BATCH_SIZE = 32
train_loader, test_loader = create_nmnist_dataloaders(BATCH_SIZE, num_workers=2, time_window=10000)

x, y = next(iter(train_loader))
print(f"data shape: {x.shape}")

multi_metric_builder = MultiMetricBuilder()
multi_metric_builder.add_metric("loss", lambda o: o.loss)
multi_metric_builder.add_metric("acc", lambda o: o.acc)
train_metrics: MultiMetric = multi_metric_builder.set_tag("train").build()
test_metrics: MultiMetric = multi_metric_builder.set_tag("test").build()

# neuron and simulation parameters
spike_grad = surrogate.atan()
beta = 0.5
#  Initialize Network

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 12, 5),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 32, 5),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*5*5, 10),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        )
    
    # this time, we won't return membrane as we don't need it
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        spk_rec = []
        utils.reset(self.layers)  # resets hidden states for all LIF neurons in net

        for step in range(data.size(0)):  # data.size(0) = number of time steps
            spk_out, mem_out = self.layers(data[step])
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)


net = Net().to(device)
optimizer = torch.optim.AdamW(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = 5
num_batches = len(train_loader)
num_iters = num_batches

@dataclass
class Output(OutputBase):
    loss: float
    acc: float

# training loop
# took 52 min on rtx 4060, achieved ~95% acc
with tqdm.trange(num_epochs) as pbar:
    for _ in pbar:
        net.train()
        for i, (data, targets) in enumerate(iter(train_loader)):
            data = data.to(device)
            targets = targets.to(device)

            spk_rec = net(data)
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            acc = SF.accuracy_rate(spk_rec, targets)
            output = Output(loss_val.item(), acc.item())

            train_metrics.accumulate(output)

            del output

            avg_batch_loss = sum(train_metrics["loss"]._buffer) / (i+1)
            pbar.set_postfix(batch=f"{i+1}/{len(train_loader)}", loss="%.3e" % avg_batch_loss, acc="%.3e" % acc.item())

            # This will end training after 50 iterations by default
            if i == num_iters:
                break
        train_metrics.compute()

        with torch.no_grad():
            net.eval()
            for i, (data, targets) in enumerate(iter(test_loader)):
                data = data.to(device)
                targets = targets.to(device)

                net.train()
                spk_rec = net(data)
                loss_val = loss_fn(spk_rec, targets)

                # Store loss history for future plotting
                acc = SF.accuracy_rate(spk_rec, targets)
                output = Output(loss_val.item(), acc.item())

                test_metrics.accumulate(output)

                del output

                avg_batch_loss = sum(test_metrics["loss"]._buffer) / (i+1)
                pbar.set_postfix(batch=f"{i+1}/{len(test_loader)}", loss="%.3e" % avg_batch_loss, acc="%.3e" % acc.item())

                # This will end training after 50 iterations by default
                if i == num_iters:
                    break
            test_metrics.compute()

os.makedirs("./bin/bptt", exist_ok=True)
dill.dump(net, open("./bin/bptt/net.pkl", "wb"))
dill.dump(train_metrics, open("./bin/bptt/train_metrics.pkl", "wb"))
dill.dump(test_metrics, open("./bin/bptt/test_metrics.pkl", "wb"))
