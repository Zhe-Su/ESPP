* EchoSpike Predictive Plasticity (ESPP)
This repository accompanies the work "EchoSpike Predictive Plasticity: A Novel Local Online Learning Rule for Spiking Neural Networks". It implements a fully local, online-compatible learning rule for spiking neural networks (SNNs) using snnTorch LIF neurons and manual gradient assignments, avoiding backprop-through-time.

** Highlights
- Local predictive plasticity rule with no global error backpropagation
- Online and offline variants of the rule
- Multiple recurrent wiring schemes (dense, stacked, full, deep-transition)
- Experiments on SHD, NMNIST, and Poisson-coded MNIST (PMNIST)

** Repository Structure
- `main.py`: Entry point to train EchoSpike on SHD, NMNIST, or PMNIST
- `model.py`: EchoSpike network (`EchoSpike`) and layer (`EchoSpike_layer`), plus simple readouts (`simple_out`)
- `utils.py`: Training loop, evaluation helpers, readout training utilities
- `data.py`: Dataset loaders and a class-wise sampling loader
- `final_results/`, `summaries/`: Example plots and HTML summaries used in the paper
- `analyze_*.ipynb`, `plots_for_paper.ipynb`: Analysis and plotting notebooks

** Installation
Requires Python 3.9+ recommended.

1. Create and activate a virtual environment (optional but recommended):
#+begin_src bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
#+end_src

2. Install dependencies. Install PyTorch per your platform/CUDA from the official instructions, then the rest:
#+begin_src bash
# Install PyTorch (choose the right command from the official selector)
# See: https://pytorch.org/get-started/locally/

pip install snntorch tonic torchvision>=0.15 tqdm numpy scipy scikit-learn
#+end_src

Notes:
- `torchvision.transforms.v2` is used; ensure `torchvision>=0.15`.
- On Windows/CPU-only, install the CPU build of PyTorch from the selector linked above.

** Datasets
This repo supports three datasets:

- SHD (Spiking Heidelberg Digits): used as default in `main.py`.
- NMNIST (Neuromorphic MNIST): loaded via `tonic` and framed into time bins.
- PMNIST (Poisson-coded MNIST): generated from `torchvision` MNIST using Poisson rate coding.

*** SHD
`data.load_SHD` expects preprocessed tensors at:
- `./data/SHD/shd_train_x.torch`, `./data/SHD/shd_train_y.torch`
- `./data/SHD/shd_test_x.torch`,  `./data/SHD/shd_test_y.torch`

Each `*_x.torch` has shape `[num_samples, n_time_bins, 700]` (700 channels), and `*_y.torch` contains labels of shape `[num_samples]`.

To generate these from `tonic` (example, may take time and disk space):
#+begin_src python
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
#+end_src

*** NMNIST
Loaded via `tonic` and framed to `n_time_bins`. The loader supports optional train splitting. Requires internet for the first download.

*** PMNIST (Poisson-coded MNIST)
Generated with `torchvision` MNIST and snnTorch rate coding into `n_time_bins`.

** How to Train
By default, `main.py` is set to SHD with online learning and a 4-layer hidden stack of 450 neurons each.

1. Ensure the `models/` directory exists (for checkpoints and final models):
#+begin_src python
import os
os.makedirs('models', exist_ok=True)
#+end_src

2. Run training:
#+begin_src bash
python main.py
#+end_src

3. To change the dataset or hyperparameters, edit the `Args` class in `main.py`:
#+begin_src python
class Args:
    def __init__(self):
        self.model_name = 'test'
        self.dataset = 'shd'           # 'shd' | 'nmnist' | 'pmnist'
        self.online = True
        self.device = 'cpu'            # or 'cuda'
        self.recurrency_type = 'dense' # 'none' | 'stacked' | 'full' | 'dt' | 'dense'
        self.lr = 1e-4
        self.epochs = 1000
        self.augment = True
        self.batch_size = 128
        self.n_hidden = 4 * [450]
        # dataset-specific fields (c_y, inp_thr, n_inputs, n_outputs, n_time_bins, beta)
#+end_src

Trained artifacts are saved to `models/{model_name}.pt`, a loss history to `models/{model_name}_loss_hist.pt`, and the arguments to `models/{model_name}_args.pkl`.

** Configuration Reference
Key fields in `Args` (dataset-dependent defaults are set in `main.py`):
- `model_name`: Run identifier used for filenames
- `dataset`: `'shd'`, `'nmnist'`, or `'pmnist'`
- `online`: Online learning if True; offline predictive/contrastive if False
- `device`: `'cpu'` or `'cuda'`
- `recurrency_type`: `'none'`, `'stacked'`, `'full'`, `'dt'`, `'dense'`
- `lr`, `epochs`, `augment`, `batch_size`
- `n_hidden`: list of hidden sizes per layer
- `beta`: leak parameter; can be float, list per layer, or negative for heterogeneous betas
- `n_time_bins`, `n_inputs`, `n_outputs`: set per dataset
- `c_y`, `inp_thr`: online rule coefficients and input-activity threshold

** Model and Learning Rule
The network is a stack of `EchoSpike_layer`s. Each layer contains a linear projection and a snnTorch `Leaky` LIF neuron. The forward pass runs under `torch.no_grad()` and maintains input and spike traces across time and phases.

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

** Evaluation and Readouts
The repo includes helpers to train simple readouts from input and from each hidden layer:
- `utils.train_out_proj_fast`: trains linear readouts with SGD
- `utils.train_out_proj_closed_form`: trains readouts via least-squares or ridge
- `utils.get_accuracy`: computes accuracy for input and layer-wise readouts

Minimal evaluation example (after training on SHD):
#+begin_src python
import torch
import pickle
from model import EchoSpike
from data import load_SHD
from utils import get_accuracy

# Load args and model
with open('models/test_args.pkl', 'rb') as f:  # created by main.py
    args = pickle.load(f)
net = EchoSpike(args.n_inputs, args.n_hidden, c_y=args.c_y, beta=args.beta,
                device=args.device, recurrency_type=args.recurrency_type,
                n_time_steps=args.n_time_bins, online=args.online, inp_thr=args.inp_thr)
net.load_state_dict(torch.load('models/test.pt', map_location=args.device))

train_loader, test_loader = load_SHD(batch_size=128)

# Example: train closed-form readouts from input and each layer (SHD: 700 inputs, 20 classes)
from utils import get_samples, train_out_proj_closed_form
snn_samples, targets = get_samples(net, train_loader, args.n_hidden, args.device)
out_projs = train_out_proj_closed_form(args, snn_samples, targets, cat=False, ridge=True)

accs, pred_matrix = get_accuracy(net, out_projs, test_loader, args.device, cat=False)
print(accs)
#+end_src

Note: Some utilities assume SHD dimensions (700 inputs, 20 classes). Adapt if you use other datasets.

** Results
Precomputed plots and pickled metrics are in `final_results/`. Notebooks `analyze_shd.ipynb` and `analyze_mnist.ipynb` reproduce key figures.

** Citation
If you use this code, please cite the EchoSpike paper. https://arxiv.org/abs/2405.13976

** License
This project is released under the terms of the license in `LICENSE`.

** Acknowledgements
- Built on [snnTorch](https://snntorch.readthedocs.io/)
- Uses [Tonic](https://tonic.readthedocs.io/) for neuromorphic datasets and framing
