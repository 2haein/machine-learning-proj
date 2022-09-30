# Classification for multiple classes using softmax and cross entropy

## 1. Baseline notebook code

- [assignment_04.ipynb](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/04/assignment_04.ipynb)
 
## 2. Data

- [assignment_04_data.npz](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/04/assignment_04_data.npz)
- input data consists of `training` data and `testing` data
- `training` data is used for training neural network
- `testing` data is used for validating the trained neural network
- both `training` and `testing` data have the same structure that consists of a pair of image $`x`$ and label $`y`$
- $`x`$ denotes a list of 2-dimensional matrices and $`y`$ denotes a list of vectors
- $`x[0, :, :]`$ represents the first image and $`x[1, :, :]`$ represents the second image
- $`y[0]`$ represents the label for $`x[0, :, :]`$ and $`y[1]`$ represents the label for $`x[1, :, :]`$
- $`x`$ represent images for digits 0, 1, 2, 3, 4
- $`y`$ represent labels for the digits 0, 1, 2, 3, 4
- label is defined by one-hot encoding scheme as follows:
  - label for the digit 0 is defined by $`(1, 0, 0, 0, 0)`$
  - label for the digit 1 is defined by $`(0, 1, 0, 0, 0)`$
  - label for the digit 2 is defined by $`(0, 0, 1, 0, 0)`$
  - label for the digit 3 is defined by $`(0, 0, 0, 1, 0)`$
  - label for the digit 4 is defined by $`(0, 0, 0, 0, 1)`$

## 3. Neural Network

- neural network $`f_w(x)`$ consists of a linear layer followed by the `softmax` activation function 
- neural network $`f_w(x)`$ for input $`x`$ is defined by:
```math
f_w(x) = \sigma( w^T x ),
```
where $`w`$ denotes weights in the fully connected linear layer and $`\sigma`$ denotes softmax (normalized exponential) function defined by:
```math
\sigma(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}
```
where $`z = (z_1, z_2, \cdots, z_K) \in \mathbb{R}^K`$ and $`z_i = w_i^T x`$
- output $`h = f_w(x)`$ of the neural network $`f_w(x)`$ for input $`x`$ is considered as prediction value $`h = (h_1, h_2, \cdots, h_K) \in \mathbb{R}^K`$ where $`h_i = \sigma(z)_i`$ for each class
- label of input $`x`$ is determined by:
```math
l(x) = \arg\max_j \sigma(z)_j
```
where $`l(x)`$ denotes a label function that determines the class of $`x`$

## 4. Loss function

- loss function is defined for a set of training data $`\{ (x_i, y_i) \}_{i=1}^n`$ by the cross entropy between $`h = (h_1, h_2, \cdots, h_K) \in \mathbb{R}^K`$ and $`y = (y_1, y_2, \cdots, y_K) \in \mathbb{R}^K`$:
```math
\mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} \ell_i(w),
```
where $`\ell_i(w)`$ denotes loss for a pair of data $`x_i`$ and label $`y_i`$ as defined by:
```math
\ell_i(w) = - \sum_{j = 1}^K y_{i, j} \log(h_{i, j})
```

## 5. Gradient

- gradient descent step is given as follows:
```math
w^{(t+1)} \coloneqq w^{(t)} - \eta \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(w),
```
where the gradient is defined by:
```math
\begin{align}
\frac{\partial \ell}{\partial w_k} & = - y_k \frac{\partial \log(h_k)}{\partial w_k} - \sum_{j \neq k} y_j \frac{\partial \log(h_j)}{\partial w_k} \\
\frac{\partial \log(h_k)}{\partial w_k} & =  h_k ( 1 - h_k ) x\\
\frac{\partial \log(h_j)}{\partial w_k}  & = - (h_j \, h_k) x
\end{align}
```
and we have:
```math
\frac{\partial \ell}{\partial w_k} = (h_k - y_k) x
```

## 6. Optimization by Gradient Descent

- gradient descent step is given as follows:
```math
w^{(t+1)} \coloneqq w^{(t)} - \eta \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(w),
```
where $`\eta`$ denotes the learning rate 

## 7. GitHub history

- `commit` should be made at least 10 times
- the message for each `commit` should be informative with respect to the modification of the code
- the GitHub history should effectively indicate the pregress of coding

---

# [Submission]

1. [x] jupyter notebook file (ipynb) for the complete code (filename should be `01-whatever-you-like.ipynb`)
2. [x] PDF file exported from the complete jupyter notebook file (filename should be `02-whatever-you-like.pdf`)
3. [x] PDF file for the GitHub history of the jupyter notebook file (filename should be `03-whatever-you-like.pdf`)
