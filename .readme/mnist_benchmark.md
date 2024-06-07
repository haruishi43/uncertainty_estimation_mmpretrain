# MNIST Benchmark

Hyperparams:
- lenet5
- 30 epochs
- bs=128
- SGD with 0.01

| Method | Loss | Evidence Function | Accuracy |
| ------ | ---- | ----------------- | -------- |
| Base   |      |                   | 98.99    |
| EDL    | SSE  | relu              | 85.93    |
| EDL    | SSE  | exp               | 98.89    |
| EDL    | SSE  | softplus          | 98.03    |
| EDL    | NLL  | exp               | 99.22    |
| EDL    | CE   | exp               | 99.27    |
| EDL    | CE   | exp-tanh          | 99.33    |
| EDL    | CE   | softplus          | 98.54    |

## SSE+ReLU:

I could not consistently reproduce SSE loss:
- Heavily depends on the random seed
  - Most of the time it converges to 60~70% accuracy
  - But it sometimes converges to 98% accuracy
- It is sensitive to hyperparameters
- Horrible performance for uncertainty estimation

| Loss | Evidence Function | Optim          | Seed | Accuracy |
| ---- | ----------------- | -------------- | ---- | -------- |
| SSE  | relu              | SGD (0.01)     | 3407 | 85.93    |
| SSE  | relu              | SGD (0.01)     | 0    | 97.15    |
| SSE  | relu              | Adam (0.001)   | 0    | 86.88    |
| SSE  | relu              | AdamW (0.002)  | 0    | 98.63    |

When testing another open source implementation with `relu` activation, it does worse than my implementation:
- [1](https://github.com/teddykoker/evidential-learning-pytorch)
  - it only scored 30~50% accuracy...
- [2](https://github.com/dougbrion/pytorch-classification-uncertainty/tree/master)
  - I don't know about their accuracy, but their implementation seems to be more stable

With some modifications to LeNet5 as in [2](https://github.com/dougbrion/pytorch-classification-uncertainty/tree/master), I was able to obtain 97~98% accuracy with `relu` activation reliably. The LeNet5 they use, however, adds more channels to all layers, which is not the original LeNet5.
The reproduced uncertainty estimation values are also similar to the original paper.


# MNIST Benchmark #2 (Adam Optimizer)

Hyperparams:
- lenet5 (modified)
- 30 epochs
- bs=1000
- Adam with 0.001

| Method | Loss | Evidence Function | Accuracy |
| ------ | ---- | ----------------- | -------- |
| Base   |      |                   | 99.12    |
| EDL    | SSE  | relu              | 97.51    |
| EDL    | SSE  | softplus          | 98.56    |
| EDL    | SSE  | exp               | 97.45    |
| EDL    | SSE  | exp-tanh          | 99.05    |
| EDL    | NLL  | exp               | 98.45    |
| EDL    | CE   | exp               | 97.97    |
| EDL    | CE   | exp-tanh          | 99.28    |
| EDL    | CE   | softplus          | 98.95    |
| REDL   | SSE  | softplus          | 98.49    |

I think these values are more desirable than the previous benchmark.
