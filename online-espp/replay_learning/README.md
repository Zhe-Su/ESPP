# replay learning

## How to run:
In root directory run: 

```
PYTHONPATH=. python replay_learning/espp_online/train_espp_ff_online.py
```

This trains a feedforward network on NMNIST, creates T-SNE plots, and stores the weihgts and metrices in a `checkpoint.pkl`. 

## Code Structure

Everything moved into `src/` directory. 

All espp components are implemented in `espp_online/espp_online.py` with layers from `util/blocks.py`. Each layer is wrapped by an `OnlineESPPLayer` that is considered its own network trained by an `OnlineESPPLayerTrainer`. The entire training is then orchastrated by an `OnlineESPPTrainer`.

This structure allows to track metrics per each layer and it assures that pytorch does not calculate any gradients since the layers are never connected to a torch Module.

The network's parameters can be changed in the `espp_online/train...py` files in the `espp_layer_func` functions. This includes using the `replay buffer` and selecting the learning rules `"gated_mi"` and `"espp"`. 