# Unsupervised image denoising with a regularization term using Total Variation

- pytorch
- neural network architecture
- stochastic gradient descent
- weight decay
- mean squared error
- psnr
- total variation
- learning rate annealing scheme

## 1. Baseline notebook code

- [assignment_08.ipynb](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/08/assignment_08.ipynb)

## 2. Data

- [assignment_08_data.npz](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/08/assignment_08_data.npz)
- data consists of `training` data and `testing` data
- `training` data is used for training a neural network
- `testing` data is used for validating the trained neural network
- both `training` and `testing` data have the same structure that consists of a pair of clean image $`I`$ and its corresponding noisy image $`J`$
- images $`I(x, y)`$ and $`J(x, y)`$ are 2-dimensional matrices (gray-scale images)
- noisy image is obtained by the summation of the original image and random noise sampled from the normal distribution with 0 mean

## 3. Neural Network

- construct a neural network in the form of auto-encoder that consists of encoder and decoder
- typical encoder consists of layers including convolution, pooling, activation, batch normalization
- typical decoder consists of layers including upsampling, activation, batch normalization
- typical activation function for the output of the network is Sigmoid
- use the following functions:
  - `Conv2d`
  - `MaxPool2d`
  - `ReLU`
  - `BatchNorm2d`
  - `Upsample`
  - `ConvTranspose2d`
  - `Sigmoid`
- construct a neural network architecture in such a way that best testing accuracy can be achieved

## 4. Loss

- loss consists of the data fidelity term and the regularization term
- the data fidelity term $`\rho(w; J)`$ is defined by the mean squared error:
```math
  \rho(w; J) = \frac{1}{n} \sum_{x, y} (J(x, y) - \hat{J}(x, y))^2
```
where $`n`$ is the length of the data $`J`$, and $`\hat{J}`$ is a prediction of the neural network for input $`J`$
- the regularization term $`\gamma(w; J)`$ is defined by the total variation:
```math 
  \gamma(w; J) = \frac{1}{n} \sum_{x, y} \left( \left \vert \frac{\partial}{\partial x} \hat{J}(x, y) \right \vert + \left \vert \frac{\partial}{\partial y} \hat{J}(x, y) \right \vert \right) 
```
where $`n`$ is the length of the data $`J`$, and $`\hat{J}`$ is a prediction of the neural network for input $`J`$
- partial derivative of $`\hat{J}`$ with respect to the $`x`$-direction is defined by:
```math
  \frac{\partial}{\partial x} \hat{J}(x, y) = \hat{J}(x+1, y) - \hat{J}(x, y)
```
- partial derivative of $`\hat{J}`$ with respect to the $`y`$-direction is defined by:
```math
  \frac{\partial}{\partial y} \hat{J}(x, y) = \hat{J}(x, y+1) - \hat{J}(x, y)
```
- use the neumann boundary condition:
```math 
\hat{J}(x+1, y) = \hat{J}(x, y)
```
along the boundary at the computation of $`\frac{\partial}{\partial x} \hat{J}(x, y)`$
```math 
\hat{J}(x, y+1) = \hat{J}(x, y)
```
along the boundary at the computation of $`\frac{\partial}{\partial y} \hat{J}(x, y)`$
- the objective function $`\ell_i(w)`$ for each pair of data $`(I_i, J_i)`$ is defined by:
```math
  \ell_i(w) = \rho(w; J_i) + \alpha \cdot \gamma(w; J_i)
```
- the total loss $`\mathcal{L}(w)`$ is defined by:
```math 
\mathcal{L}(w) = \frac{1}{m} \sum_{i=1}^m \ell_i(w)
```
where $`m`$ is the number of data

## 5. Optimization

- use stochastic gradient descent (sgd) optimizer in the pytorch library
- use any learning rate annealing scheme that includes Adam, AdaGrad and RMSProp
- you can use weight decay

## 6. Training

- training aims to determine the model parameters of the neural network and its associated loss function is minimized using the training data

## 7. Testing

- testing aims to validate the generality of the trained neural network using the testing data

## 8. Initialization

- initialization of the weights of the neural network architecture in such a way that best testing accuracy can be achieved 

## 9. Evaluation

- use PSNR for the evaluation of the performance
- PSNR is computed by $`PSNR = 10 * \log_{10} \left( \frac{MAX(IMAGE)^2}{MSE} \right)`$
- set $`MAX(IMAGE) = 1`$ for our dataset
- $`MSE(x, y) = \frac{1}{n} \| x - y \|_2^2`$ where $`n`$ is the size of data $`x`$ and $`y`$
 
## 10. Hyper-parameters

- determine the followings in such a way that best testing accuracy can be achieved
  - number of epochs
  - size of mini-batch
  - learning-rate
  - weight-decay

## 11. Grading

- the scores are given by the ranking of the final testing accuracy within correct answers
  - rank 01 - 10 : 5
  - rank 11 - 20 : 4
  - rank 21 - 30 : 3
  - rank 31 -  : 2

## 12. GitHub history

- `commit` should be made at least 10 times
- the message for each `commit` should be informative with respect to the modification of the code
- the GitHub history should effectively indicate the pregress of coding

# [Submission]

1. [x] jupyter notebook file (ipynb) for the complete code (filename should be `01-whatever-you-like.ipynb`)
2. [x] PDF file exported from the complete jupyter notebook file (filename should be `02-whatever-you-like.pdf`)
3. [x] PDF file for the GitHub history of the jupyter notebook file (filename should be `03-whatever-you-like.pdf`)