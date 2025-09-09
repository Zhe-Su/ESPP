# EchoSpike Predictive Plasticity (ESPP)
This repository accompanies the work "EchoSpike Predictive Plasticity: A Novel Local Online Learning Rule for Spiking Neural Networks". It implements a fully local, online-compatible learning rule for spiking neural networks (SNNs) using snnTorch LIF neurons and manual gradient assignments, avoiding backprop-through-time.

## Highlights
- Local predictive plasticity rule with no global error backpropagation
- Online and offline variants of the rule
- Multiple recurrent wiring schemes (dense, stacked, full, deep-transition)
- Experiments on SHD, NMNIST, and Poisson-coded MNIST (PMNIST)

## Repository Structure
- `src/`: Core source code directory
  - `main.py`: Entry point
  - `trainer.py`: Trainer class
  - `model.py`: EchoSpike network (`EchoSpike`) and layer (`EchoSpike_layer`)
  - `utils.py`: Evaluation helpers, readout training utilities
  - `data.py`: Dataset loaders and class-wise sampling loader
- `config/`: Hydra configuration files
  - `config.yaml`: Main configuration file with model and training parameters
  - `dataset/`: Dataset-specific configurations (SHD, NMNIST, PMNIST)
- `notebooks/`: Jupyter notebooks for analysis and visualization
  - `analyze_shd.ipynb`, `analyze_mnist.ipynb`: Dataset-specific analysis
  - `plots_for_paper.ipynb`: Generate publication figures
- `outputs/`: Experiments outputs and logs
- `data/`: Dataset storage (HDF5 files for preprocessed data)
- `final_results/`, `summaries/`: Precomputed results and HTML summaries from paper
- `environment.yaml`: Conda environment specification

## Installation

1. Clone the repository and navigate to the project directory:
```bash
git clone <repository-url>
cd ESPP
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate ESPP
```

The environment includes all necessary dependencies:
- PyTorch with CUDA support
- snnTorch for spiking neural networks
- Tonic for neuromorphic datasets
- Hydra for configuration management
- Scientific computing stack (NumPy, SciPy, scikit-learn)
- Visualization tools (Matplotlib, Seaborn)
- Jupyter for interactive analysis

For manual installation or different platforms, see the dependency list in `environment.yaml`.

## Datasets
This repo supports three datasets:

- SHD (Spiking Heidelberg Digits): used as default in `main.py`.
- NMNIST (Neuromorphic MNIST): loaded via `tonic` and framed into time bins.
- PMNIST (Poisson-coded MNIST): generated from `torchvision` MNIST using Poisson rate coding.

### SHD
For SHD (Spiking Heidelberg Digits), you can either:

**Option 1: Use preprocessed data (recommended):**
Download preprocessed HDF5 files and place them in the `data/` directory. The training script will automatically load them.

**Option 2: Generate from Tonic:**
`data.load_SHD` can generate tensors from Tonic. This expects preprocessed tensors at:
- `./data/SHD/shd_train_x.torch`, `./data/SHD/shd_train_y.torch`
- `./data/SHD/shd_test_x.torch`,  `./data/SHD/shd_test_y.torch`

Each `*_x.torch` has shape `[num_samples, n_time_bins, 700]` (700 channels), and `*_y.torch` contains labels of shape `[num_samples]`.

To generate these from `tonic` (example, may take time and disk space):
```python
import torch
import tonic
from tonic import transforms

sensor_size = tonic.datasets.SHD.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=100)

train = tonic.datasets.SHD(save_to='./data', train=True, transform=frame_transform)
test  = tonic.datasets.SHD(save_to='./data', train=False, transform=frame_transform)

def to_tensor(ds):
    X = []
    y = []
    for i in range(len(ds)):
        frames, label = ds[i]          # frames: [n_time_bins, 700]
        X.append(torch.tensor(frames))
        y.append(label)
    X = torch.stack(X)                 # [N, T, 700]
    y = torch.tensor(y)
    return X, y

train_x, train_y = to_tensor(train)
test_x, test_y = to_tensor(test)

torch.save(train_x, './data/SHD/shd_train_x.torch')
torch.save(train_y, './data/SHD/shd_train_y.torch')
torch.save(test_x, './data/SHD/shd_test_x.torch')
torch.save(test_y, './data/SHD/shd_test_y.torch')
```

### NMNIST
Loaded via `tonic` and framed to `n_time_bins`. The loader supports optional train splitting. Requires internet for the first download.

### PMNIST (Poisson-coded MNIST)
Generated with `torchvision` MNIST and snnTorch rate coding into `n_time_bins`.

## How to Train

The project uses Hydra for configuration management. All training parameters are specified in YAML files under `config/`.

### Quick Start Examples

1. **Train on NMNIST with default settings:**
```bash
cd src
python main.py
```

2. **Train on SHD dataset:**
```bash
cd src
python main.py dataset=shd
```

3. **Train on PMNIST dataset:**
```bash
cd src
python main.py dataset=pmnist
```

4. **Use offline learning mode:**
```bash
cd src
python main.py online=false
```

5. **Train with custom parameters:**
```bash
cd src
python main.py dataset=shd lr=1e-3 batch_size=64 epochs=500
```

6. **Train with different recurrency types:**
```bash
cd src
python main.py recurrency_type=full          # Full recurrency
python main.py recurrency_type=stacked       # Stacked recurrency
python main.py recurrency_type=none          # Feedforward only
```

### Output Structure

Training outputs are organized by Hydra in the `outputs/` directory:
```
outputs/
├── YYYY-MM-DD/
│   └── HH-MM-SS/
│       ├── src/                     # Complete copy of source code
│       │   └── ...
│       ├── config/                  # Complete copy of configuration files
│       │   ├── config.yaml
│       │   └── ...
│       ├── checkpoints/             # Model checkpoints during training
│       │   └── ...
│       ├── media/                   # Generated plots and visualizations
│       │   └── ...

```

## Configuration Reference

The project uses Hydra for hierarchical configuration management. Configuration files are located in `config/`:

### Main Configuration (`config/config.yaml`)
```yaml
# Dataset configuration (overridden by dataset-specific configs)
defaults:
  - dataset: nmnist  # or 'shd', 'pmnist'
  - _self_

# Model and training parameters
model_name: 'test'                    # Experiment identifier
online: true                          # Online vs offline learning
device: 'cuda'                        # 'cpu' or 'cuda'
recurrency_type: 'dense'              # 'none', 'stacked', 'full', 'dt', 'dense'
lr: 1e-4                             # Learning rate
epochs: 1000                         # Training epochs
augment: false                       # Data augmentation
batch_size: 128                      # Batch size
n_hidden: [200, 200, 200]           # Hidden layer sizes
seed: 123                           # Random seed
```

### Dataset-Specific Configurations
Located in `config/dataset/`:

**SHD (`shd.yaml`)**:
- `n_inputs: 700`, `n_outputs: 20`, `n_time_bins: 100`
- `beta: [0.94, 0.96, 0.98, 1.0]` (leak parameters per layer)
- `c_y: [1.5, -1.5]` (online) / `c_y_offline: [8e-4, -4e-4]` (offline)

**NMNIST (`nmnist.yaml`)**:
- `n_inputs: 68`, `n_outputs: 10`, `n_time_bins: 300`

**PMNIST (`pmnist.yaml`)**:
- `n_inputs: 784`, `n_outputs: 10`, `n_time_bins: 100`

### Override Parameters
You can override any configuration parameter from the command line:
```bash
# Override single parameters
python main.py lr=1e-3 batch_size=64

# Override nested parameters
python main.py n_hidden=[512,512,512,512]

# Override dataset and its parameters
python main.py dataset=shd dataset.n_time_bins=200
```

## Model and Learning Rule
The network is a stack of `EchoSpike_layers`. Each layer contains a linear projection and a snnTorch `Leaky` LIF neuron. The forward pass runs under `torch.no_grad()` and maintains input and spike traces across time and phases.

Learning is local and toggles between predictive/contrastive phases via a broadcasting factor `bf ∈ {1, -1}`:
- Offline mode: accumulate outer products across the time dimension and apply a weight update after each predictive+contrastive pair during `reset()`.
- Online mode: update per time step using a sign of a running local loss and a simple surrogate gradient on the membrane potential, gated by input activity.

Recurrent wiring options (`recurrency_type`):
- `none`: purely feedforward
- `stacked`: concatenate previous layer state to the next layer input
- `full`: concatenate all hidden states and inputs to every layer
- `dt`: deep-transition (feeds last layer back to the first)
- `dense`: concatenate input and all previous layer outputs progressively

See `model.py` for exact input dimensionalities per scheme.

## Evaluation and Readouts

The repo includes helpers to train simple readouts from input and from each hidden layer:
- `utils.train_out_proj_fast`: trains linear readouts with SGD
- `utils.train_out_proj_closed_form`: trains readouts via least-squares or ridge
- `utils.get_accuracy`: computes accuracy for input and layer-wise readouts

### Evaluation Examples

After training a model, you can evaluate it in several ways:

1. **Load and evaluate a trained model:**
```python
import torch
import pickle
from model import EchoSpike
from data import load_SHD
from utils import get_accuracy

# Load from Hydra output (recommended)
model_path = "outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/checkpoint.pt"
config_path = "outputs/YYYY-MM-DD/HH-MM-SS/config/config.yaml"

# Initialize and load model
net = EchoSpike(args.n_inputs, args.n_hidden, c_y=args.c_y, beta=args.beta,
                device=args.device, recurrency_type=args.recurrency_type,
                n_time_steps=args.n_time_bins, online=args.online, inp_thr=args.inp_thr)
net.load_state_dict(torch.load(model_path, map_location=args.device))

# Load dataset
train_loader, test_loader = load_SHD(batch_size=128)

# Train readouts and evaluate
from utils import get_samples, train_out_proj_closed_form
snn_samples, targets = get_samples(net, train_loader, args.n_hidden, args.device)
out_projs = train_out_proj_closed_form(args, snn_samples, targets, cat=False, ridge=True)

accs, pred_matrix = get_accuracy(net, out_projs, test_loader, args.device, cat=False)
print("Layer-wise accuracies:", accs)
```

2. **Interactive analysis with Jupyter notebooks:**
```bash
cd notebooks
jupyter lab analyze_shd.ipynb        # SHD analysis
jupyter lab analyze_mnist.ipynb      # NMNIST/PMNIST analysis
jupyter lab plots_for_paper.ipynb    # Generate publication figures
```

Note: Some utilities assume SHD dimensions (700 inputs, 20 classes). Adapt if you use other datasets.

## Results

Precomputed plots and pickled metrics are in `final_results/`. Interactive analysis notebooks in `notebooks/` reproduce key figures:

- `analyze_shd.ipynb`: SHD dataset analysis and visualization
- `analyze_mnist.ipynb`: NMNIST and PMNIST analysis
- `plots_for_paper.ipynb`: Generate publication-quality figures

Training outputs and logs are automatically organized in `outputs/` by Hydra with timestamps for easy experiment tracking.

## Citation
If you use this code, please cite the EchoSpike paper. https://arxiv.org/abs/2405.13976

## License
This project is released under the terms of the license in `LICENSE`.

## Acknowledgements
- Built on [snnTorch](https://snntorch.readthedocs.io/)
- Uses [Tonic](https://tonic.readthedocs.io/) for neuromorphic datasets and framing
