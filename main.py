import multiprocessing

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import util.data_loader
import util.tensor
from modules.attention import TransformerSR
from util.database import GelbooruDatabase

if __name__ == "__main__":
    batch_size = 1
    is_training = False

    # Retrieve data
    posts = GelbooruDatabase().get_posts()

    # Count tags to create loss weights
    tag_counts = {}
    for post in posts:
        for tag in post["tags"].split():
            tag_counts[tag] = 1 if tag not in tag_counts else tag_counts[tag] + 1
    
    # Sort tags by name and discard uncommon tags  
    tags = sorted(
        [(tag, count) for tag, count in tag_counts.items() if count / len(posts) > 0.01],
        key = lambda x: x[0]
    )

    # Loss weights of each index
    loss_weights = [len(posts) / count for _, count in tags]
    weight_sum = sum(loss_weights)
    loss_weights = [weight / weight_sum for weight in loss_weights]

    # Tag name of each index
    tags = [tag for tag, _ in tags]

    # Start producer for loading images
    if is_training:
        batch_queue = multiprocessing.Queue(50)
        multiprocessing.Process(target=util.data_loader.load_data_loop, args=(posts, tags, batch_size, batch_queue)).start()

    # Model initialization
    model = TransformerSR(factor=4, residual_groups=[], num_blocks=8, dim=256, window_size=4, num_heads=16).cuda()
    model.load_state_dict(torch.load("result/models/super_resolution.pt"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Number of parameters: {sum(param.numel() for param in model.parameters())}")

    # Test
    if not is_training:
        model.eval()
        H, W = 224, 128

        inputs = util.tensor.load_tensor("result/ayanami.png").cuda()
        save_image(util.tensor.resize_and_crop(inputs, H * 4, W * 4), f"result/hr.png")

        inputs = torch.stack([util.tensor.resize_and_crop(inputs, H, W)])

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = model(inputs)

        save_image(inputs[0], f"result/lr.png")
        save_image(F.interpolate(output, (H * 4, W * 4), mode="bicubic")[0], f"result/sr.png")

    # Sobel filter
    gx = torch.stack([torch.Tensor([[[1,0,-1],[2,0,-2],[1,0,-1]]])] * 3).cuda()
    gy = torch.stack([torch.Tensor([[[1,2,1],[0,0,0],[-1,-2,-1]]])] * 3).cuda()

    # Train
    i = 1
    while is_training:
        batch = batch_queue.get()
        inputs = batch["inputs"].cuda()

        inputs_lr = util.tensor.random_downsample(1/4)(inputs)

        with torch.cuda.amp.autocast():
            edge_map = (
                F.conv2d(inputs, weight=gx, padding=1,groups=3) ** 2 +
                F.conv2d(inputs, weight=gy, padding=1,groups=3) ** 2
            ) ** 0.5

            output = model(inputs_lr)
            loss = (torch.abs(output - inputs) * edge_map.detach()).sum() / edge_map.sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        print(f"\r#{i:<6d} - loss: {loss.item():8.4f}",end="")
        if (i % 1000 == 0):
            print(f" saved as result/images/{i}.png - model saved")
            save_image(inputs_lr[0], f"result/images/{i}_lr.png")
            save_image(inputs[0], f"result/images/{i}_hr.png")
            save_image(output[0], f"result/images/{i}.png")
            torch.save(model.state_dict(), "result/models/super_resolution.pt")
        i += 1