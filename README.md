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
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/52ed7733-6e59-4b40-9066-fb60cd611a29"></td>
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/6a5785ff-7248-4b49-a71c-358dd5afe5b0"></td>
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/a69509fc-b452-433d-b681-9b8391496b73"></td>
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/f9a1ce6c-c9c1-479f-a158-7c1eb3fbc9ee"></td>
      <td><img width=100 src="https://github.com/Craap/models/assets/110075485/9867267d-4b4f-46f5-9c57-4e82361b0348"></td>
      <td><img width=100 src=https://github.com/Craap/models/assets/110075485/2650a7c8-abec-495a-a889-9b865b84bb86""></td>
    </tr>
  </table>

  <h3>TODO</h3>

  - Due to downsampling, image is shifted to the top left, fix this
  - Implement GAN loss
  - Improve it somehow
