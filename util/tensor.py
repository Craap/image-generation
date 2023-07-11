import random

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None

def load_tensor(image_path: str) -> Tensor | None:
    """
    Returns 3-channel tensor that is loaded from `image_path`
    or `None` if loading failed

    Also changes image format to RGBA
    """
    
    try:
        image = Image.open(image_path)
        if image.format == "PNG" and image.mode != "RGBA":
            image = image.convert("RGBA")
            image.save(image_path)
            
        return transforms.ToTensor()(image.convert('RGB'))
    except:
        return None
    
def resize_and_crop(tensor: Tensor, height: int, width: int):
    """
    Returns resized `tensor` such that one of its sides
    matches the corresponding `height` or `width`
    and the other side is longer, then center crop the result to `(height, width)`
    """

    if tensor.shape[-2] / height > tensor.shape[-1] / width:
        new_height = int(tensor.shape[-2] * width / tensor.shape[-1])
        tensor = transforms.Resize((new_height, width), antialias=True)(tensor)
    else:
        new_width = int(tensor.shape[-1] * height / tensor.shape[-2])
        tensor = transforms.Resize((height, new_width), antialias=True)(tensor)

    return transforms.CenterCrop((height, width))(tensor)

def center_crop(tensor: Tensor, divisible_by: int) -> Tensor:
    """Returns the largest center cropped `tensor` whose sides are divisible by `divisible_by`"""
    
    width = tensor.shape[-1]
    height = tensor.shape[-2]

    while width % divisible_by != 0:
        width -= 1
    
    while height % divisible_by != 0:
        height -= 1

    return transforms.CenterCrop((height, width))(tensor)

def random_crop(tensor: Tensor, size) -> Tensor:
    return transforms.RandomCrop(size)(tensor)

def random_flip(tensor: Tensor) -> Tensor:
    return transforms.RandomVerticalFlip()(transforms.RandomHorizontalFlip()(tensor))

def resize(tensor: Tensor, max_area: int) -> Tensor:
    """Returns `tensor` with the same aspect ratio such that its area is at most `max_area`"""
    
    image_area = tensor.shape[-1] * tensor.shape[-2]
    if image_area <= max_area:
        return tensor
    
    scale = (image_area / max_area) ** 0.5
    return transforms.Resize(
        (int(tensor.shape[-2] / scale), int(tensor.shape[-1] / scale)),
        antialias=True
    )(tensor)

def random_downsample(scale_factor: float):
    """Returns downsample function with randomly chosen method"""
    mode = random.choice(["bilinear", "bicubic", "area"])
    antialias = False if mode == "area" else True
    return lambda tensor: F.interpolate(tensor, scale_factor=scale_factor, mode=mode, antialias=antialias)

def random_noise(variance: float):
    """Returns gaussian noise function with randomly chosen variance"""
    return lambda tensor: tensor + torch.randn_like(tensor) * (random.random() * variance) ** 0.5

def random_blur(kernel_sizes: list | tuple):
    """Returns gaussian blur function with randomly chosen kernel size"""
    return transforms.GaussianBlur(random.choice(kernel_sizes))
