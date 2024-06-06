# Uncertainty Estimation with Evidential Deep Learning

Experiments for Evidential Deep Learning (EDL)

The original EDL paper can be accessed at [arXiv](http://arxiv.org/abs/1806.01768).

The goals of this project are:
- to reproduce the results of the paper
- for me to understand how EDL works and the capabilities
- to adapt EDL for other datasets

The project introduces:
- modular implementation compatible with `mmpretrain`, enabling easy integration with other models and datasets
- various evidence functions (e.g. softplus, exponential, etc.)
- various loss functions (e.g. SSE, NLL, etc.)
- novel formulations (e.g. R-EDL)

Future work:
- Implement other uncertainty estimation methods (e.g. MC dropout, Ensembles, DDU, etc...)
- Add uncertainty metrics to quantify the methods
- Benchmark

## Installation

Install `torch` (with gpu-support).
Tested for `cu117` version of `torch`:
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install dependencies:
```bash
pip install -r requirements.txt
pip install -U openmim
mim install "mmpretrain>=1.0.0rc8"
```

## MNIST Example

### Training classifiers

```python
# softmax baseline
python tools/train.py configs/edl_mnist/default_lenet5_mnist.py

# edl
python tools/train.py configs/edl_mnist/edl_lenet5_mnist.py
```

### Visualization

Checkout `notebooks/exp_edl_mnist.ipynb` to visualize the results.


## Experiments

### Experiments in the paper

Used `softplus` as the evidence function.

| Experiment | Softmax  | Evidential Deep Learning |
| ---------- | -------- | -------- |
| Rotate "1" | ![alt text](.readme/rotate_deterministic_model.png) | ![alt text](.readme/rotate_edl_model.png) |
| Classify "1" | ![alt text](.readme/one_deterministic_model.png) | ![alt text](.readme/one_edl_model.png) |
| Classify "Yoda" | ![alt text](.readme/yoda_deterministic_model.png) | ![alt text](.readme/yoda_edl_model.png) |

I've noticed that EDL is very sensitive to how it is trained.
For example, when I used the Adam optimizer for training, the model accuracy improves, but the uncertainty estimates are not as good as when I used SGD.

### Different Evidence Functions

| Evidence Function | Rotated One Experiment |
| ---------- | -------- |
| `relu(x)` | ![alt text](.readme/rotate_relu_model.png) |
| `exp(x)` (clamped) | ![alt text](.readme/rotate_exp_model.png) |
| `exp(tanh(x) / tau)` | ![alt text](.readme/rotate_exptanh_model.png) |

### Different Loss Functions

Implemented `SSE`, `NLL`, and `CE` loss functions for classification task.
Follwing the original paper, the default loss function used in the project is `SSE`.
However, I've experimented with the other loss functions as well.

| Loss Function | Rotated One Experiment |
| ---------- | -------- |
| `SSE` | ![alt text](.readme/rotate_edl_model.png) |
| `NLL` | ![alt text](.readme/rotate_nll_model.png) |
| `CE` | ![alt text](.readme/rotate_ce_model.png) |
| `Relaxed SSE` | ![alt text](.readme/rotate_redl_model.png) |

Note that choosing the loss function also requires choosing the right evidence function.
For example, it is said that `SSE` works well with `softplus` (and maybe `relu`), while `CE` works well with `exp`.
Recent applications to CV tasks have shown that `CE` works well with `exp(tanh(x) / tau)`.


### CIFAR-5 Experiments

Instead of LeNet5, I used ResNet18 for CIFAR-5 experiments.

Uncertainty thresholded accuracy plot:
![alt text](.readme/uncertainty_accuracy_plot.png)

Empirical Cumulative Distribution Function (ECDF) of the uncertainty estimates:
![alt text](.readme/ecdf.png)

## Acknowledgement

- [evidential-learning-pytorch](https://github.com/teddykoker/evidential-learning-pytorch)
  - Referenced the implementation for Evidential Deep Learning which I ported to `mmpretrain`.
- [pytorch-classification-uncertainty](https://github.com/dougbrion/pytorch-classification-uncertainty)
  - Borrowed experimental setup and visualization code.
