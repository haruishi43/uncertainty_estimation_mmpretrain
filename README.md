# Evidential Deep Learning

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

| Experiment | Softmax  | Evidentail |
| ---------- | -------- | -------- |
| Rotate "1" | ![alt text](.readme/rotate_deterministic_model.png) | ![alt text](.readme/rotate_edl_model.png) |
| Classify "1" | ![alt text](.readme/one_deterministic_model.png) | ![alt text](.readme/one_edl_model.png) |
| Classify "Yoda" | ![alt text](.readme/yoda_deterministic_model.png) | ![alt text](.readme/yoda_edl_model.png) |


## Acknowledgement

- [evidential-learning-pytorch](https://github.com/teddykoker/evidential-learning-pytorch)
  - Referenced the implementation for Evidential Deep Learning which I ported to `mmpretrain`.
- [pytorch-classification-uncertainty](https://github.com/dougbrion/pytorch-classification-uncertainty)
  - Borrowed experimental setup and visualization code.
