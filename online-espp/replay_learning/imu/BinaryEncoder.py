import torch
import torch.nn as nn

class Binary_Encoder(nn.Module):
    """ Binary encoding scheme based on the algorithm from the paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10675361"""
    
    def __init__(self, n_bits=10):
        """
        Args:
            n_bits: Number of bits used for the binary encoding (resolution).
            n_channels: Number of channels in the input signal.
        """
        super().__init__()
        self.n_bits = n_bits
    
    def forward(self, x):
        batch_size, time_steps, channels = x.size()
        
        # Ensure signal values are in range [0, 1]
        x = torch.clamp(x, 0, 1)
        
        # Initialize an empty spike train with shape (batch_size, time_steps, n_bits, n_channels)
        spike_train = torch.zeros(batch_size, time_steps, self.n_bits, channels)

        # Loop over each sample in the batch
        for b in range(batch_size):
            for t in range(time_steps):
                for c in range(channels):
                    # Get the signal value for the current data point
                    vsignal = x[b, t, c]
                    
                    # Initialize the bit decision threshold
                    bit_decision = 0.5
                    
                    # Perform binary encoding based on the number of bits
                    for bit in range(self.n_bits):
                        if vsignal >= bit_decision:
                            # Set a spike in the spike train
                            spike_train[b, t, bit, c] = 1
                            # Subtract the decision threshold from the signal value
                            vsignal -= bit_decision
                        
                        # Halve the decision threshold for the next bit
                        bit_decision *= 0.5
                
        return spike_train
