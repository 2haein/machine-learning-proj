# Image Segmentation using an unsupervised learning framework

- data augmentation: flip, rotation, crop, etc (input data, ground truth for evaluation)
- neural network architecture: skip-connection, U-net
- loss: unsupervised loss based on a bi-partitioning model
- accuracy measure: IoU (Intersection over Union)
- optimiser: sgd, Adam, AdaGrad, RMSProp
- bi-partitioning image model: K-means clustering

## 1. Baseline notebook code

- [assignment_10.ipynb](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/10/assignment_10.ipynb)

## 2. Data

- [assignment_10_data.npz](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/10/assignment_10_data.npz)
- data consists of `training` data and `testing` data
- `training` data is used for training a neural network
- `testing` data is used for validating the trained neural network
- both `training` and `testing` data have the same structure that consists of a pair of input image $`I`$ and its corresponding segmentation mask $`M`$
- images $`I(x, y)`$ and masks $`M(x, y)`$ are 2-dimensional matrices (gray-scale images)
- input image is obtained by the summation of the binary image and random noise sampled from the normal distribution with 0 mean
- you can use data augmentation for training using flip, rotate, crop and so on

## 3. Neural Network

- construct a neural network in the form of auto-encoder that consists of encoder and decoder
- activation function for the output of the network should be Sigmoid
- construct a neural network architecture in such a way that best testing accuracy can be achieved
- you can use skip-connections similar to U-net architecture

## 4. Loss

- total loss consists of the data fidelity term and the regularization term

### 4.1. Data fidelity term

- the data fidelity term is defined by a piecewise constant image model:

```math
I(x, y) = a * \hat{I}(x, y) + b * (1 - \hat{I}(x, y)) + \eta(x, y) 
```

where $`\eta`$ is a noise process that is assumed to follow a normal distribution with mean 0

- the data fidelity $`\rho(w; I)`$ is defined by a bi-partitioning function based on prediction $`\hat{I}`$ as follows:

```math
\rho(w; I) = \frac{1}{n} \sum_{x, y} \left\{ \hat{I}(x, y) * (I(x, y) - a)^2 + (1 - \hat{I}(x, y)) * (I(x, y) - b)^2 \right\}
```

where $`n`$ is the number of elements in $`I`$ and $`\hat{I} = f_w(I)`$ is a prediction of $`I`$ with a neural network $`f_w`$ parameterised by $`w`$

- $`a \in \mathbb{R}`$ and $`b \in \mathbb{R}`$ are estimates for the inside and the outside of segmenting region represented by $`\hat{I}`$, respectively
- estimates $`a`$ and $`b`$ are obtained by:

```math
a = \frac{\sum_{x, y} I(x, y) * \hat{I}(x, y)}{\sum_{x, y} \hat{I}(x, y)}, \qquad 
b = \frac{\sum_{x, y} I(x, y) * (1 - \hat{I}(x, y))}{\sum_{x, y} (1 - \hat{I}(x, y))},
```

where $`*`$ indicates an elementwise multiplication operator



### 4.2. Regularization term

- the regularization term $`\gamma(w; I)`$ is defined by the total variation of the prediction $`\hat{I}`$:

```math 
\gamma(w; I) = \frac{1}{n} \sum_{x, y} \left( \left \vert \frac{\partial}{\partial x} \hat{I}(x, y) \right \vert + \left \vert \frac{\partial}{\partial y} \hat{I}(x, y) \right \vert \right) 
```

where $`n`$ is the number of elements in $`I`$ and $`\hat{I} = f_w(I)`$ is a prediction of $`I`$ with a neural network $`f_w`$ parameterised by $`w`$

- partial derivative of $`\hat{J}`$ with respect to the $`x`$-direction is defined by:

```math
\frac{\partial}{\partial x} \hat{I}(x, y) = \hat{I}(x+1, y) - \hat{I}(x, y)
```

- partial derivative of $`\hat{J}`$ with respect to the $`y`$-direction is defined by:

```math
\frac{\partial}{\partial y} \hat{I}(x, y) = \hat{I}(x, y+1) - \hat{I}(x, y)
```

- use the neumann boundary condition:

```math 
\hat{I}(x+1, y) = \hat{I}(x, y)
```

along the boundary at the computation of $`\frac{\partial}{\partial x} \hat{I}(x, y)`$

```math 
\hat{I}(x, y+1) = \hat{I}(x, y)
```

along the boundary at the computation of $`\frac{\partial}{\partial y} \hat{I}(x, y)`$

### 4.3. Total loss

- the objective function $`\ell_i(w)`$ for data $`I_i`$ is defined by:

```math
\ell_i(w) = \rho(w; I_i) + \alpha \cdot \gamma(w; I_i)
```

- the total loss $`\mathcal{L}(w)`$ is defined by:

```math 
\mathcal{L}(w) = \frac{1}{m} \sum_{i=1}^m \ell_i(w)
```

where $`m`$ is the number of data

## 5. Optimization

- you can use an optimiser using sgd, Adam, AdaGrad, RMSProp and so on

## 6. Training

- training aims to determine the model parameters of the neural network and its associated loss function is minimized using the training data

## 7. Testing

- testing aims to validate the generality of the trained neural network using the testing data

## 8. Initialization

- initialization of the weights of the neural network architecture in such a way that best testing accuracy can be achieved 

## 9. Evaluation

- segmentation result for input image $`I`$ is obtained in the form of binary image by taking a threshold $`0.5`$ with respect to the inference $`\hat{I} = f_w(I)`$

- consider both $`\hat{I}`$ and $`1 - \hat{I}`$ for the prediction since there is no difference between foreground and background in the prediction while there is such difference in mask $`M`$

- use Intersection over Union (IoU) for the evaluation of the performance

- `IoU` is computed by the ratio of intersection and union of the prediction and the mask:

```math
\textrm{IoU} = \frac{\textrm{Area of Intersection}}{\textrm{Area of Union}}
```

## 10. Hyper-parameters

- determine the hyper-parameters in such a way that best testing accuracy can be achieved

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

## [Submission]

1. [x] jupyter notebook file (ipynb) for the complete code (filename should be `01-whatever-you-like.ipynb`)
2. [x] PDF file exported from the complete jupyter notebook file (filename should be `02-whatever-you-like.pdf`)
3. [x] PDF file for the GitHub history of the jupyter notebook file (filename should be `03-whatever-you-like.pdf`)
