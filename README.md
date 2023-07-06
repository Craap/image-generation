This is a repository for my experiments on image generation and related stuff

<h1>Requirements</h1>

- <a href="https://www.python.org/downloads/release/python-3106/">Python 3</a> (specifically, I use python 3.10.6)
- <a href="https://pytorch.org/get-started/locally/">Torch</a>

<h1>Super resolution</h1>

  <h3>Architecture</h3>
  
  The architecture is similar to <a href="https://arxiv.org/pdf/2204.03645v1.pdf">DaViT</a> (probably), except I don't split channel attention into groups.

  Basically the model is an alternating sequence of window attention and group attention, with pixel shuffle upscaling at the end

  I also tried to use a lot of residual connections between the layers, following most SR models, but in the end decided to omit them for simplicity
  
  <h3>Data</h3>
  
  Dataset is 240 handpicked images from the internet
  
  Samples are random cropped to create more data

  To obtain LR samples, the HR samples are downsampled using a uniformly chosen method between bilinear, bicubic, and area
  
  I also tried random blurring and random noise too for a few training steps, but it didn't seem to produce good results, and random cropping is already enough to not overfit
  
  <h3>Loss function</h3>
  
  For the first half of training, only L1 loss is used
  
  I tried to make the loss more focused on high frequency details (edges) by multiplying the L1 loss (before reduction) with an edge map, this produced much better results

  The edge map is created by applying a sobel filter to the ground truth image

  <h3>Result</h3>
  See code for hyper parameters
  
  Trained for 1 million steps, pretrained weights <a href="https://huggingface.co/Craap/models/blob/main/transformerSR_b8_d256_w8_h16.pt">here</a>

  Qualitative comparisons with pretrained SwinIR models:
  <table>
    <tr>
      <th>LR</th>
      <th>HR</th>
      <th>SwinIR-L-GAN</th>
      <th>SwinIR-L-PSNR</th>
      <th>SwinIR-S</th>
      <th>Mine</th>
    </tr>
    <tr>
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/e8ffb5b8-d49a-461a-bc7a-b98d006fd974"></td>
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/9c859f13-f959-47ca-9c05-ac138b43baa7"></td>
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/59bf3dc6-6b47-4c02-af19-38dfca66826b"></td>
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/0e4a932d-4223-4b0d-a5e6-487a3d6732ad"></td>
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/93c294dc-57ab-4e92-8a2e-fb2581b1d8c7"></td>
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/1064d86a-30ae-4867-974d-4f4b34e1ad6e"></td>
    </tr>
  </table>

  Sample image is generated with Stable Diffusion, using my personal trained LoRA

  <a href="https://github.com/JingyunLiang/SwinIR">SwinIR</a> is used for comparisons here because other models have too many files and I'm lazy

  I used pretrained weights for SwinIR instead of training on the same dataset, so that might be a little unfair for SwinIR

  All pretrained weights for Swin-M produce NaNs for me, so there is no result

  <h3>TODO</h3>

  - Due to downsampling, image is shifted to the top left, fix this
  - Implement GAN loss
  - Improve it somehow
