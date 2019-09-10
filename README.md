# VDSR in Tensorflow

Tensorflow implementation of **Accurate Image Super-Resolution Using Very Deep Convolutional Networks**.

*GT* | *Bicubic* | *SRCNN* | *VDSR*
:---: | :---: | :---: | :---: |
<img src = 'figs/intro_gt.png'> | <img src = 'figs/intro_bicubic.png'> | <img src = 'figs/intro_srcnn.png'> | <img src = 'figs/intro_vdsr.png'>

## Implementation Details

### Network Architecture

| *Layer*      | *# of layers* |   *Filter size* | *Input, output channel* | *Activation Function* |
| :----------: | :-----------: | :-------------: | :---------------------: | :-------------------: |
| Input Layer  | 1             | 3 x 3           | (1, 64)                 | ReLU                  |
| Hidden Layer | 18            | 3 x 3           | (64, 64)                | ReLU                  |
| Output Layer | 1             | 3 x 3           | (64, 1)                 | -                     |

### Loss function

- $$ Loss(W)=\frac{1}{2}||r-f(x)||^{2} $$
- Mean Squared Error loss
- Residual learning 

### Regularization

- $$ reg(W)=\frac{\lambda}{2}\sum_{w \in W} {||w||^{2}} $$
- L2 Regularization
- $ \lambda $: 0.0001

### Optimization

- Weight initialization: He method
- Bias initialization: Zero initialize
- AdamOptimizer
    - Learning rate: 0.0001
    - Epoch: 80
    - Batch size: 64
    - Iteration per epoch: 11579
    - No learning rate decaying, gradient clipping are used

### Training Dataset

- 291 images dataset with data augmentation (rotate or flip) is used
- Data augmentation
    - Downsize with (1.0, 0.7, 0.5) scales
    - Rotate (0, 90, 180, 270) degrees
    - Flip left-right
- More than 700,000 pairs are generated (up to 20GB)

## Installation

```bash
git clone https://github.com/jinsuyoo/VDSR-Tensorflow.git
```

## Requirements

You will need the following to run the above:
- Tensorflow-gpu
- Python3, Numpy, Pillow, h5py, tqdm

To install quickly, use `requirements.txt`. Example usage:
```bash
pip install -r requirements.txt
```
Note that we run the code with Windows 10, Tensorflow-gpu 1.13.1, CUDA 10.0, cuDNN v7.6.0 

## Documentation

To pre-process the train and test dataset, you need to execute the Matlab code.

Generating training data takes about half an hour, up to 20GB.

The pre-processed test data with Set5 and Set14 is provided.

### Training VDSR
Use `main.py` to train the network. Run `python main.py` to view the training process. Training takes 80 hours on a NVIDIA GeForce GTX 1050. Example usage:
```bash
# Quick training
python main.py

# Example usage
python main.py --epoch=40 --layer_depth=10 --l2_lambda=1e-3 --optimizer=momentum

# Usage
python main.py [-h] [--epoch EPOCH] [--batch_size BATCH_SIZE] 
               [--layer_depth LAYER_DEPTH] [--starter_learning_rate STARTER_LEARNING_RATE] [--decay_epochs DECAY_EPOCHS] [--optimizer OPTIMIZER] [--momentum MOMENTUM] [--grad_clip GRAD_CLIP] [--l2_regularization] [--l2_lambda L2_LAMBDA] [--train_dataset TRAIN_DATASET] [--valid_dataset VALID_DATASET]
               
optional arguments:
  -h, --help                Show this help message and exit
  --epoch                   Number of epoch for training (Default: 80)
  --batch_size              Batch size for training (Default: 64)
  --layer_depth             Depth of the network layer (Default: 20)
  --starter_learning_rate   Starter learning rate for training (Default: 1e-4)
  --decay_epochs            epoch for learning rate

```

### Testing VDSR
Also use `main.py` to test the network. Pretrained-model with 91-image training dataset and up-scaling factor 3 is given. Example usage:
```bash
# Quick testing
python main.py --is_training=False \
    --use_pretrained=True

# Example usage
python main.py --is_training=False \
    --use_pretrained=True \
    --test_dataset=YOUR_DATASET \
    --scale=4
```
  
Please note that if you want to train or test with your own dataset, you need to execute the Matlab code with your own dataset first :)

## Results

### Performance curve for the residual network

- Validated under **Set5** dataset with scale factor 2, 3 and 4
- x-axis: epoch, y-axis: PSNR value

<img src = 'figs/psnr_x2.png'> | <img src = 'figs/psnr_x3.png'> | <img src = 'figs/psnr_x4.png'> 
:---: | :---: | :---: |
*scale 2* | *scale 3* | *scale 4* 

### The average results of PSNR (dB) value

*Dataset*    | *Scale* | *Bicubic* | *VDSR*  | *VDSR-Tensorflow* 
:----------: | :-----: | :-------: | :-----: | :---------------: 
**Set5**     | x2      | dB   | 37.53dB |
**Set5**     | x3      | dB   | 33.66dB |
**Set5**     | x4      | dB   | 31.35dB |
**Set14**    | x2      | dB   | 33.03dB |
**Set14**    | x3      | dB   | 29.77dB |
**Set14**    | x4      | dB   | 28.01dB |
**B100**     | x2      | dB   | 31.90dB |
**B100**     | x3      | dB   | 28.82dB |
**B100**     | x4      | dB   | 27.29dB |
**Urban100** | x2      | dB   | 30.76dB |
**Urban100** | x3      | dB   | 27.14dB |
**Urban100** | x4      | dB   | 25.18dB |


### Some of the result images

*GT* | *Bicubic* | *VDSR* 
:---: | :---: | :---: |
<img src = 'figs/result1_gt.png'> | <img src = 'figs/result1_bicubic.png'> | <img src = 'figs/result1_vdsr.png'> 
<img src = 'figs/result2_gt.png'> | <img src = 'figs/result2_bicubic.png'> | <img src = 'figs/result2_vdsr.png'> 
<img src = 'figs/result3_gt.png'> | <img src = 'figs/result3_bicubic.png'> | <img src = 'figs/result3_vdsr.png'> 
<img src = 'figs/result4_gt.png'> | <img src = 'figs/result4_bicubic.png'> | <img src = 'figs/result4_vdsr.png'> 


## References

- [Official Website][1]
    - We referred to the original Matlab and Caffe code.

- [jinsuyoo/SRCNN-Tensorflow][2]
    - Our implementation of SRCNN.
 
[data]: https://drive.google.com/file/d/1yvQYDYKCrTNxtvkOAHpTFOapEDyji0RR/view?usp=sharing
[1]: https://cv.snu.ac.kr/research/VDSR/
[2]: https://github.com/jinsuyoo/SRCNN-Tensorflow