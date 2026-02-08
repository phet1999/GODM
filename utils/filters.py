import torch
from torch import nn


def get_factors(n):
    f = list(
        set(
            factor
            for i in range(2, int(n**0.5) + 1)
            if n % i == 0
            for factor in (i, n // i)
        )
    )
    f.sort()
    f.append(n)
    f = [1] + f
    return f


# downsampling through moving average
class MovingAvgTime(nn.Module):
    """
    Moving average block to highlight the trend of time series, only for factors kernal size
    """

    def __init__(self, kernel_size, seq_length: int, stride=1):
        super(MovingAvgTime, self).__init__()
        self.kernel_size = kernel_size
        self.seq_length = seq_length
        K = torch.zeros(seq_length, int((seq_length - kernel_size) / stride + 1))
        start = 0
        for i in range(K.shape[1]):
            end = start + kernel_size
            K[start:end, i] = 1 / kernel_size
            start += stride
        K = K.unsqueeze(0)
        mode = "nearest-exact" if stride == 1 else "linear"
        self.K = (
            torch.nn.functional.interpolate(K, size=seq_length, mode=mode).squeeze().T
        )

        # self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor):
        # print(x.shape)
        assert x.shape[1] == self.seq_length
        
        x = self.K.to(x.device) @ x
        return x

