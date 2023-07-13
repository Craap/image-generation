This is a repository for my experiments on image generation and related stuff

<h1>Requirements</h1>

- <a href="https://www.python.org/downloads/release/python-3106/">Python 3</a> (specifically, I use python 3.10.6)
- <a href="https://pytorch.org/get-started/locally/">Torch</a>

<h1>Super resolution</h1>

  <h2>Architecture</h2>
  
  The architecture is similar to <a href="https://arxiv.org/pdf/2204.03645v1.pdf">DaViT</a> (probably, haven't checked the code), except I don't split channel attention into groups.

  Basically the model is a single convolution applied to the input, followed by a sequence of alternating window attention and group attention, followed by a single convolution and pixel shuffle upscaling at the end
  
  <h2>Data</h2>
  
  Dataset is 240 handpicked images from the internet
  
  <h2>Loss function</h2>
  
  Loss function is L1 loss multiplied with an edge map
  
  The edge map is obtained by applying a sobel filter to the HR image

  <h2>Hyper parameters</h2>
  
  Model:

  - Attention window size: 4
  - Number of attention blocks: 16 (8 window attention and 8 channel attention blocks)
  - Hidden dimension: 256
  - Number of attention heads: 16
  
  Other:

  - Learning rate: constant 3e-5
  - Training steps: 2 million
  - Batch size: 1
  - Images are scaled to a max area of 1 million pixels, then random cropped to 256 x 256 and random flipped to obtain the HR image, LR image is 64 x 64
  

  <h2>Result</h2>
  
  Pretrained weights <a href="https://huggingface.co/Craap/image-generation/blob/main/transformerSR_b8_d256_w4_h16.pt">here</a>

  Quantitatively, my model has worse PSNR than <a href="https://github.com/JingyunLiang/SwinIR">SwinIR</a> (and others), but qualitatively I think it looks better, at least on anime style images

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
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/25aa2832-15a6-4b28-8eaf-62ad2a8290c9"></td>
    </tr>
  </table>

  Sample image is generated with Stable Diffusion, using my personal trained LoRA

  SwinIR is used for comparisons here because other models have too many files and I'm lazy

  I used pretrained weights for SwinIR instead of training on the same dataset, so that might be a little unfair for SwinIR

  All pretrained weights for Swin-M produce NaNs for me, so there is no result

  <h2>TODO</h2>

  - Implement GAN loss
  - Improve it somehow
