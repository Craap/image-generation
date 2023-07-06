import random
from multiprocessing import Queue

import torch
from torch import Tensor

from .tensor import load_tensor, random_crop, random_flip, resize_and_crop


def get_aspect_ratios(divisible_by: int, min_side: int, max_area: int):
    """
    Returns a set of `(H, W)` where:
    - `H * W <= max_area`
    - `H, W >= min_side`
    - `H` and `W` are divisible by `divisible_by`

    This is done by first incrementing `min_side` until it is divisible by `divisible_by`,
    then the other side is `max_area / min_side` decremented until divisible by `divisible_by`.
    Once this is done we obtain a pair `(H, W)`, increase `min_side` by `divisible_by` and repeat
    the process to obtain all valid pairs `(H, W)`
    """

    ratio_set = set()

    height = min_side
    while height % divisible_by != 0:
        height += 1

    width = max_area // height
    while width % divisible_by != 0:
        width -= 1

    # Create aspect ratios with height <= width
    while height <= width:
        ratio_set.add((height, width))

        height += divisible_by
        width = max_area // height
        while width % divisible_by != 0:
            width -= 1

    # Create aspect ratios with width < height
    for i, j in list(ratio_set):
        ratio_set.add((j, i))

    return ratio_set

def make_buckets(posts: list, aspect_ratios: dict) -> dict:
    """
    Returns a mapping `(H, W) -> list(post)` with `(H, W)` from `aspect_ratios`
    and `post` from `posts`,
    where each `post` is assigned to the `(H, W)` with the closest aspect ratio
    """

    buckets = {}
    for ratio in aspect_ratios:
        buckets[ratio] = []

    for post in posts:
        ratio = post["height"] / post["width"]
        buckets[min(aspect_ratios, key = lambda x: (ratio - x[0] / x[1]) ** 2)].append(post)

    return buckets

def load_resize_crop(image_path: str, height: int, width: int) -> Tensor:
    tensor = load_tensor(image_path)
    if tensor is None:
        return None
    return resize_and_crop(tensor, height, width)

def load_data_loop(posts: list, tags: list, batch_size: int, batch_queue: Queue):
    # Create buckets and discard buckets with less images than batch size
    buckets = make_buckets(posts, get_aspect_ratios(64, 128, 1000000))
    buckets = {k: v for k, v in buckets.items() if len(v) >= batch_size}
    bucket_weights = [len(images) for images in buckets.values()]

    while True:
        inputs = []
        labels = []

        # Get random bucket
        (height, width), post_list = random.choices(list(buckets.items()), bucket_weights)[0]

        # Get random batch from bucket
        for post in random.sample(post_list, batch_size):
            # Load image
            input_tensor = load_resize_crop(f"data/images/{post['image']}", height, width)
            if input_tensor is None:
                inputs = []
                labels = []
                break

            # Load tags
            tags_of_post = post["tags"].split()
            label_tensor = torch.Tensor([1 if tag in tags_of_post else 0 for tag in tags])

            inputs.append(input_tensor)
            labels.append(label_tensor)

        if len(inputs) > 0 and len(inputs) == len(labels):
            inputs = torch.stack(inputs)

            # Make 10 256x256 batches out of each batch
            for _ in range(10):
                batch = {
                    "inputs": random_flip(random_crop(inputs), 256),
                    "labels": torch.stack(labels),
                }
                batch_queue.put(batch)
