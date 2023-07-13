import torch
import torch.nn.functional as F
from torch import Tensor, nn


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """Partitions (B, C, H, W) into (B, num_windows, C, `window_size`, `window_size`)"""
    B, C, H, W = x.shape
    num_windows = H * W // window_size ** 2
    return (
        x.reshape(B, C, H // window_size, window_size, W // window_size, window_size)
         .permute(0, 1, 2, 4, 3, 5)
         .reshape(B, C, num_windows, window_size, window_size)
         .permute(0, 2, 1, 3, 4)
    )

def window_unpartition(x: Tensor, H: int, W: int) -> Tensor:
    """Unpartitions (B, num_windows, C, `window_size`, `window_size`) into (B, C, H, W)"""
    B, _, C, window_size, _ = x.shape
    return (
        x.permute(0, 2, 1, 3, 4)
         .reshape(B, C, H // window_size, W // window_size, window_size, window_size)
         .permute(0, 1, 2, 4, 3, 5)
         .reshape(B, C, H, W)
    )

def spatial_flatten(x: Tensor) -> Tensor:
    """Flattens (*, C, H, W) into (*, H * W, C)"""
    return x.flatten(-2).transpose(-2, -1)
    
def spatial_unflatten(x: Tensor, H: int, W: int) -> Tensor:
    """Unflattens (*, H * W, C) into (*, C, H, W)"""
    return x.transpose(-2, -1).unflatten(-1, (H, W))

class MultiHeadAttention(nn.Module):
    """
    Performs multi head attention on the input sequence,
    can also transpose inputs before performing attention
    - Input: (*, L, `dim`)
    - Output: (*, L, `dim`)
    """

    def __init__(self, dim: int, transposed: bool, num_heads: int = 8) -> None:
        """
        Parameters:
        - `dim`: number of channels
        - `transposed`: if `True`, computes `(qT x k x vT)T` instead of `q x kT x v`
        - `num_heads`: number of attention heads
        """

        super().__init__()
        self.dim = dim
        self.transposed = transposed
        self.num_heads = num_heads
        self.project_q = nn.Linear(dim, dim)
        self.project_k = nn.Linear(dim, dim)
        self.project_v = nn.Linear(dim, dim)
        self.project_out = nn.Linear(dim, dim)
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        L, C = q.shape[-2], q.shape[-1]
        head_dim = C // self.num_heads

        q = (
            self.project_q(q)
            .unflatten(-1, (self.num_heads, head_dim))
            .transpose(-2, -3)
        )
        k = (
            self.project_k(k)
            .unflatten(-1, (self.num_heads, head_dim))
            .transpose(-2, -3)
        )
        v = (
            self.project_v(v)
            .unflatten(-1, (self.num_heads, head_dim))
            .transpose(-2, -3)
        )

        if self.transposed:
            x = torch.matmul(
                F.softmax(
                    torch.matmul(
                        q.transpose(-2, -1) / L,
                        k / L
                    ),
                    -1
                ),
                v.transpose(-2, -1)
            ).transpose(-1, -3).transpose(-1, -2)
        else: 
            x = torch.matmul(
                F.softmax(
                    torch.matmul(
                        q / head_dim,
                        k.transpose(-2, -1) / head_dim
                    ),
                    -1
                ),
                v
            ).transpose(-2, -3)
        
        x = self.project_out(
            x.flatten(-2)
        )

        return x
    
class AttentionBlock(nn.Module):
    """
    Block of `MultiHeadAttention` > `Linear` > `Linear`
    - Input: (B, N, `dim`)
    - Output: (B, N, `dim`)
    """

    def __init__(self, dim: int, transposed: bool, window_size: int = 8, num_heads: int = 8) -> None:
        """
        Parameters:
        - `dim`: number of channels
        - `transposed`: if `True`, computes `(qT x k x vT)T` instead of `q x kT x v`
        - `window_size`: size of square window
        - `num_heads`: number of attention heads
        """

        super().__init__()
        self.window_size = window_size
        self.transposed = transposed
        self.window_attention = MultiHeadAttention(dim, transposed, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape

        if self.transposed:
            x = x.unsqueeze(1)
        else:
            x = window_partition(x, self.window_size)
        x = spatial_flatten(x)

        x_residual = x
        x = self.norm1(x)
        x = x_residual + self.window_attention(x, x, x)
        x = x + self.mlp(self.norm2(x))

        if self.transposed:
            x = spatial_unflatten(x, H, W)
            x = x.squeeze(1)
        else:
            x = spatial_unflatten(x, self.window_size, self.window_size)
            x = window_unpartition(x, H, W)

        return x
    
class SR(nn.Module):
    """
    Sequence of `AttentionBlock` with pixel shuffle upsampling at the end
    - Input: (B, `dim`, H, W)
    - Output: (B, `dim`, `H * factor`, `W * factor`)
    """
    
    def __init__(
            self,
            factor: int,
            num_blocks: int,
            dim: int,
            window_size: int,
            num_heads: tuple | list,
            in_channels: int = 3,
            out_channels: int = 3
    ) -> None:
        """
        Parameters:
        - `factor`: upscaling factor, must be a power of 2
        - `num_blocks`: number of blocks
        - `dim`: number of channels of the main path through the model
        - `window_size`: attention window size
        - `num_heads`: number of attention heads
        - `in_channels`: number of input channels (default 3)
        - `out_channels`: number of output channels (default 3)
        """
        
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            *[
                nn.Sequential(
                    AttentionBlock(dim, False, window_size, num_heads),
                    AttentionBlock(dim, True, window_size, num_heads)
                ) for _ in range(num_blocks)
            ],
            nn.Conv2d(dim, (factor ** 2) * out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(factor)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = x - 0.5
        x = self.layers(x)
        return x + 0.5
    
class Classifier(nn.Module):
    """
    Groups of `AttentionBlock` with downsampling inbetween
    and global pool into logits at the end
    - Input: (B, `dim`, H, W)
    - Output: (B, `num_classes`)
    """

    def __init__(
        self,
        groups: tuple | list,
        dims: tuple | list,
        window_size: int,
        num_heads: int,
        num_classes: int,
        in_channels: int = 3
    ) -> None:
        """
        Parameters:
        - `groups`: number of blocks in each group, before each downsample
        - `dim`: number of channels of the main path through the model
        - `window_size`: attention window size
        - `num_heads`: number of attention heads
        - `in_channels`: number of input channels (default 3)
        - `out_channels`: number of output channels (default 3)
        """
        
        super().__init__()

        self.blocks = nn.ModuleList([nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1)])
        for i in range(len(groups)):
            if i > 0:
                self.blocks.append(nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2))
            for _ in range(groups[i]):
                self.blocks += [
                    AttentionBlock(dims[i], False, window_size // 2 ** i, num_heads),
                    AttentionBlock(dims[i], True, window_size // 2 ** i, num_heads)
                ]

        self.blocks += [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dims[-1], num_classes)
        ]

    def forward(self, x: Tensor) -> Tensor:
        x = x - 0.5
        for block in self.blocks:
            x = block(x)
        return x