# Multi-layer neural networks for a binary classification

## 1. Baseline notebook code

- [assignment_03.ipynb](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/03/assignment_03.ipynb)
 
## 2. Data

- [assignment_03_data.npz](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/03/assignment_03_data.npz)

## 3. Neural Network

- neural network $`h_{u, v}(x)`$ consists of two consecutive computation units
- each computation unit consists of a fully connected linear layer followed by the `sigmoid` activation function
- neural network $`h_{u, v}(x)`$ for input $`x`$ is defined by:
```math
\begin{align}
h_{u, v}(x) &= g_v \circ f_u(x)
\end{align}
```
- the first computational unit $`f_u(x)`$ is defined by:
```math
\begin{align}
f_{u}(x) &= \sigma( u^T x )
\end{align}
```
- the second computational unit $`g_v(z)`$ is defined by:
```math
\begin{align}
g_{v}(z) &= \sigma( v^T z )\\
z &= f_u(x)
\end{align}
```
- $`u`$ denotes weights for the fully connected linear layer in the first computational unit $`f_u`$
- $`v`$ denotes weights for the fully connected linear layer in the second computational unit $`g_v`$
- $`\sigma`$ denotes the sigmoid function defined by:
```math
\sigma(z) = \frac{1}{1 + \exp(-z)}
```
- note that the derivative of sigmoid function $`\sigma(z)`$ is given by:
```math
\frac{d \, \sigma(z)}{d \, z} = \sigma(z) (1 - \sigma(z))
```
- output of the neural network $`h_{u, v}(x)`$ for input $`x`$ is considered as prediction value for the class of input $`x`$ as follows:
```math
\begin{cases}
l(x) = 0 & \colon h < 0.5 \\
l(x) = 1 & \colon h \ge 0.5,
\end{cases}
```
where $`l(x)`$ denotes a label function that determines the class of $`x`$

## 4. Loss function

- loss function over a given set of training data $`\{ (x_i, y_i) \}_{i=1}^n`$ is defined by:
```math
\mathcal{L}(u, v) = \frac{1}{n} \sum_{i=1}^{n} \ell_i(u, v),
```
where $`\ell_i(u, v)`$ denotes the loss defined by the cross entropy between the ground truth $`y_i`$ of input $`x_i`$ and its prediction $`h_{u, v}(x_i)`$ as follows:
```math
\ell_i(u, v) = - \left\{ y_i \log{(h_{u, v}(x_i))} + (1 - y_i) \log{(1 - h_{u, v}(x_i))} \right\}
```

## 5. Gradient

- gradients of the loss $`\mathcal{L}(u, v)`$ using a set of training data $`\{ (x_i, y_i) \}_{i=1}^n`$ with respect to the weights $`u`$ and $`v`$ are defined by:
```math
\begin{align}
\nabla_u \mathcal{L}(u, v) &= \frac{1}{n} \sum_{i=1}^{n} \nabla_u \ell_i(u, v),\\
\nabla_v \mathcal{L}(u, v) &= \frac{1}{n} \sum_{i=1}^{n} \nabla_v \ell_i(u, v),
\end{align}
```
- the gradients of $`\nabla _u \ell_i(u, v)`$ with respect to $`u`$ for each pair of input $`(x_i, y_i)`$ is defined by (we omit the subscript $`i`$ for ease of notation):
```math
\begin{align}
\nabla_u \ell(u, v) &= \frac{\partial \ell}{\partial h} \frac{\partial h}{\partial u} \\
&= \left( -y \frac{1}{h} + (1-y) \frac{1}{1-h} \right) \frac{\partial h}{\partial u} \\
\frac{\partial h}{\partial u_1} &= (h - y) v_1 z_1 (1-z_1) x \\
\frac{\partial h}{\partial u_2} &= (h - y) v_2 z_2 (1-z_2) x \\
\end{align} 
```
where 
```math
u =
\begin{bmatrix}
u_1 & u_2
\end{bmatrix},
v =
\begin{bmatrix}
v_1\\
v_2
\end{bmatrix},
z =
\begin{bmatrix}
z_1\\
z_2
\end{bmatrix},
```
- the gradients of $`\nabla_v \ell_i(u, v)`$ with respect to $`v`$ for each pair of input $`(x_i, y_i)`$ is defined by (we omit the subscript $`i`$ for ease of notation):
```math
\begin{align}
\nabla_v \ell(u, v) &= \frac{\partial \ell}{\partial h} \frac{\partial h}{\partial v} \\
&= \left( -y \frac{1}{h} + (1-y) \frac{1}{1-h} \right) \frac{\partial h}{\partial v} \\
\frac{\partial h}{\partial v} &= h (1 - h) z \\
\nabla_v \ell(u, v) &= (h - y) z, \quad \textrm{where } z = f_u(x)
\end{align}
```
- read the followings for the matrix differentiation:
  - chapter 6.5 Back-Propagation and Other Differentiation Algorithms in book [deep learning] in Reference at Google Classroom
  - [matrix calculus](https://towardsdatascience.com/matrix-calculus-for-data-scientists-6f0990b9c222)
  - [matrix differentiation](https://atmos.washington.edu/~dennis/MatrixCalculus.pdf)
  - [vector, matrix and torsor derivatives](http://cs231n.stanford.edu/vecDerivs.pdf)
  - [vector and matrix differentiation](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470173862.app4)

## 6. Optimization by Gradient Descent

- gradient descent step for weight $`u`$ is given as follows:
```math
\begin{align}
u^{(t+1)} & \coloneqq u^{(t)} - \eta \nabla_u \mathcal{L}(u, v)\\
& \coloneqq u^{(t)} - \eta \frac{1}{n} \sum_{i=1}^{n} \nabla_u \ell_i(u, v)\\
\end{align}
```
where $`\eta`$ denotes the learning rate 

- gradient descent step for weight $`v`$ is given as follows:
```math
\begin{align}
v^{(t+1)} & \coloneqq v^{(t)} - \eta \nabla_v \mathcal{L}(u, v)\\
& \coloneqq v^{(t)} - \eta \frac{1}{n} \sum_{i=1}^{n} \nabla_v \ell_i(u, v)\\
\end{align}
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
