import torch
from data import augment_shd
import numpy as np
from tqdm.notebook import trange


class Trainer:
    """
    A trainer class for Spiking Neural Networks (SNNs) using the EchoSpike learning algorithm.
    """
    
    def __init__(self, net, device='cpu', batch_size=1, lr=1e-5, online=False, augment=False):
        """
        Initialize the Trainer.
        
        Args:
            device (str or torch.device): Device to use for training ('cpu' or 'cuda')
            batch_size (int): Batch size for training
            lr (float): Learning rate
            online (bool): Whether to use online learning
            augment (bool): Whether to apply data augmentation
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.batch_size = batch_size
        self.lr = lr
        self.online = online
        self.augment = augment
        self.net = net.to(self.device)
    
    def train(self, trainloader, epochs, model_name, freeze=[]):
        """
        Trains a SNN.

        Args:
            net (torch.nn.Module): The neural network model to be trained.
            trainloader (torch.utils.data.DataLoader): The data loader for the training dataset.
            epochs (int): The number of epochs for training.
            model_name (str): Name of the model for saving checkpoints.
            freeze (list): List of layers to freeze during training.

        Returns:
            torch.Tensor: A tensor containing the loss history during training.
        """
        torch.set_grad_enabled(False)
        loss_hist = []
        accuracies = []
        print_interval = 100 * self.batch_size if 'mnist' in model_name else 40 * self.batch_size
        
        # training loop
        optimizer = torch.optim.SGD([{"params": par.fc.parameters(), 'lr': self.lr} for par in net.layers])
        optimizer.zero_grad()
        self.net.train()
        bf = 0
        target = [torch.randint(trainloader.num_classes, (1,)).item() for _ in range(self.batch_size)]
        spks = torch.zeros(len(self.net.layers) + 1, device=self.device)
        
        while True:
            # Train loop
            data, target = trainloader.next_item(target, contrastive=(bf == -1))
            data = data.float().to(self.device)
            if self.augment:
                data = augment_shd(data)
            target = target.to(self.device)
            sample_loss = torch.zeros(len(self.net.layers), device=self.device)

            i = 0
            for step in range(data.shape[0]):
                # iterate over time steps
                if self.online:
                    inp_activity = data[step].mean(axis=-1)
                else:
                    inp_activity = None
                spk, _, loss, grad = self.net(data[step], torch.tensor(bf, device=self.device), freeze, inp_activity=inp_activity)
                spks += torch.stack([data[step].mean(), *[sp.mean() for sp in spk]])    # to analyze nr of spks
                sample_loss += loss
                if self.online:
                    optimizer.step()
                    optimizer.zero_grad()
                i += 1

            loss_hist.append(sample_loss / data.shape[0]) 
            accuracies.append(self.net.reset(bf))

            if bf == -1 and not self.online:
                # update weights after one predictive and one contrastive batch, before weight update
                optimizer.step()
                optimizer.zero_grad()
            bf = 1 if bf != 1 else -1

            step = len(loss_hist) * self.batch_size
            epoch = step // len(trainloader)
            if step % print_interval < self.batch_size and len(loss_hist) > 1:
                # print loss and accuracy
                print(f"Epoch {epoch}, Step {step} \nEchoSpike Loss: {torch.stack(loss_hist[-print_interval//self.batch_size:]).mean(axis=0)}")
                print(f"Acc: {torch.stack(accuracies).mean(axis=0)}")
                accuracies = []
                print(f"Spks: {spks * self.batch_size / print_interval}")  # sparsity ratio
                spks = torch.zeros(len(self.net.layers) + 1, device=self.device)
            if epoch >= epochs:
                break
            if step % len(trainloader) < self.batch_size and epoch % 20 == 0:
                # save checkpoint
                current_epoch_loss = torch.stack(loss_hist[-20 * len(trainloader) // self.batch_size:]).mean().item()
                print(f'epoch loss: {current_epoch_loss}')
                torch.save(self.net.state_dict(), f'models/{model_name}_epoch{epoch}.pt')
        
        return torch.stack(loss_hist)

    def test(self, testloader):
        """
        Tests a SNN.

        Args:
            net (torch.nn.Module): The neural network model to be tested.
            testloader (torch.utils.data.DataLoader): The data loader for the test dataset.

        Returns:
            tuple: A tuple containing the following:
                - spk_history (list): A list of spike histories.
                - target_list (list): A list of target values.
                - losses (list): A list of loss values during testing.
        """
        torch.set_grad_enabled(False)
        self.net.eval()
        spk_history = []
        target_list = []
        losses = []

        bf = 0
        target = [torch.randint(testloader.num_classes, (1,)).item() for _ in range(self.batch_size)]
        for _ in trange(int(len(testloader) / self.batch_size)):
            data, target = testloader.next_item(target, contrastive=(bf == -1))
            target_list.append(target)
            data = data.float().to(self.device)
            target = target.to(self.device)
            logit_list = []
            activation_list = []
            loss_sample = torch.zeros(len(self.net.layers), device=self.device)
            for step in range(data.shape[0]):
                out_spk, _, loss, _ = self.net(data[step], torch.tensor(bf, device=self.device))
                logit_list.append(out_spk[-1])
                activation_list.append(out_spk)
                loss_sample += loss

            losses.append(loss_sample)
            spk_history.append(activation_list[0])
            for i in range(1, len(activation_list)):
                for l in range(len(spk_history[-1])):
                    spk_history[-1][l] += activation_list[i][l]
            self.net.reset(bf)
            bf = 1 if bf != 1 else -1
            # if len(losses)*batch_size > len(testloader):
            #     break
        return spk_history, target_list, losses
