# Classification for multiple classes using softmax and cross entropy

- bias
- stochastic gradient descent
- regularization - weight decay
  
## 1. Baseline notebook code

- [assignment_05.ipynb](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/05/assignment_05.ipynb)
 
## 2. Data

- [assignment_05_data.npz](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/05/assignment_05_data.npz)
- data consists of `training` data and `testing` data
- each data consist of pairs of images and their class labels

## 3. Neural Network

- neural network $`f_w(x)`$ consists of a fully connected linear layer followed by the `softmax` activation function
- consider a bias in the linear layer
- neural network $`f_w(x)`$ for input $`x`$ is defined by:
```math
f_w(x) = \sigma( w^T x ),
```
where $`w`$ denotes weights in the fully connected linear layer and $`\sigma`$ denotes softmax (normalized exponential) function defined by:
```math
\sigma(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}
```
where $`z = (z_1, z_2, \cdots, z_K) \in \mathbb{R}^K`$ for $`K`$ classes and $`z_i = w_i^T x`$ for $`i = 1, 2, \cdots, K`$
- output $`h = f_w(x)`$ of the neural network $`f_w(x)`$ for input $`x`$ is considered as prediction value $`h = (h_1, h_2, \cdots, h_K) \in \mathbb{R}^K`$ where $`h_i = \sigma(z)_i`$ for each class
- label of input $`x`$ is determined by:
```math
l(x) = \arg\max_j \sigma(z)_j
```
where $`l(x)`$ denotes a label function that determines the class of $`x`$

## 4. Loss function

- data fidelity is defined by cross entropy between  $`h = (h_1, h_2, \cdots, h_K) \in \mathbb{R}^K`$ and $`y = (y_1, y_2, \cdots, y_K) \in \mathbb{R}^K`$ for a pair of $`(x, y)`$ as follows: 
```math
\ell_i(w) = - \sum_{j = 1}^K y_{i, j} \log(h_{i, j})
```
- the total loss consists of data fidelity over mini-batch $`\beta`$ and $`L_2^2`$ regularization as follows:
```math
\mathcal{L}(w) = \frac{1}{| \beta |} \sum_{i \in \beta} \ell_i(w) + \frac{\alpha}{2} \| w \|_2^2,
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
- gradient for the regularization term is given by: 
```math
2 \alpha \frac{\partial \| w \|_2^2}{\partial w} = \alpha \cdot w
```

## 6. Optimization by Gradient Descent

- stochastic gradient descent step is given based on a mini-batch $`\beta`$ as follows:
```math
w^{(t+1)} \coloneqq w^{(t)} - \eta \left( \frac{1}{| \beta |} \sum_{i \in \beta} \nabla \ell_i(w^{(t)}) + \alpha \cdot w^{(t)}\right),
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
- use a drop the last scheme for the stochastic gradient descent
  
## 7. GitHub history

- `commit` should be made at least 10 times
- the message for each `commit` should be informative with respect to the modification of the code
- the GitHub history should effectively indicate the pregress of coding

## 8. Grading

- the scores are given by the ranking of the final testing accuracy within correct answers
  - rank 01 - 10 : 5
  - rank 11 - 25 : 4
  - rank 26 - 45 : 3
  - rank 46 -  : 2

---

# [Submission]

1. [x] jupyter notebook file (ipynb) for the complete code (filename should be `01-whatever-you-like.ipynb`)
2. [x] PDF file exported from the complete jupyter notebook file (filename should be `02-whatever-you-like.pdf`)
3. [x] PDF file for the GitHub history of the jupyter notebook file (filename should be `03-whatever-you-like.pdf`)

