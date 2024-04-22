import torch
import scipy.ndimage as nd


def get_device() -> torch.device:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def one_hot_embedding(
    labels: torch.Tensor,
    num_classes: int = 10,
) -> torch.Tensor:
    # Convert to One Hot Encoding
    y = torch.eye(num_classes).to(labels.device)
    return y[labels]


def rotate_img(x, deg: float):
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()
