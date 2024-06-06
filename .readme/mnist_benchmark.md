# MNIST Benchmark

| Method | Loss | Evidence Function | Accuracy |
| ------ | ---- | ----------------- | -------- |
| Base   |      |                   | 98.99    |
| EDL    | SSE  | relu              | 85.93    |
| EDL    | SSE  | softplus          | 98.03    |
| EDL    | NLL  | exp               | 99.22    |
| EDL    | CE   | exp               | 99.27    |
| EDL    | CE   | exp-tanh          | 99.33    |
| EDL    | CE   | softplus          | 98.54    |

# NOTE:
I could not reproduce SSE loss:
- Heavily depends on the random seed
  - Most of the time it converges to 60~70% accuracy
  - But it sometimes converges to 98% accuracy
- It is sensitive to hyperparameters

| Loss | Evidence Function | Optim          | Seed | Accuracy |
| ---- | ----------------- | -------------- | ---- | -------- |
| SSE  | relu              | SGD (0.01)     | 3407 | 85.93    |
| SSE  | relu              | SGD (0.01)     | 0    | 97.15    |
| SSE  | relu              | Adam (0.001)   | 0    | 86.88    |
| SSE  | relu              | AdamW (0.002)  | 0    | 98.63    |
