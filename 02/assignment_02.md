# Logistic regression for a binary classification

## 1. Baseline notebook code

- [assignment_02.ipynb](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/02/assignment_02.ipynb)

## 2. Data

- [assignment_02_data.npz](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/02/assignment_02_data.npz)

## 3. Neural Network

- neural network $`f_w(x)`$ consists of a linear layer followed by the `sigmoid` activation function 
- neural network $`f_w(x)`$ for input $`x`$ is defined by:
```math
f_w(x) = \sigma( w^T x ),
```
where $`w`$ denotes weights in the linear layer and $`\sigma`$ denotes sigmoid function defined by:
```math
\sigma(z) = \frac{1}{1 + \exp(-z)}
```
- output $`h = f_w(x)`$ of the neural network $`f_w(x)`$ for input $`x`$ is considered as prediction value for the class of input as follows:
```math
\begin{cases}
l(x) = 0 & \colon h < 0.5 \\
l(x) = 1 & \colon h \ge 0.5,
\end{cases}
```
where $`l(x)`$ denotes a label function that determines the class of $`x`$

## 4. Loss function

- loss function is defined by:
```math
\mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} \ell_i(w),
```
where $`\ell_i(w)`$ denotes the loss for a pair of data $`x_i`$ and label $`y_i`$ as defined by:
```math
\ell_i(w) = - \left\{ y_i \log{(f_w(x_i))} + (1 - y_i) \log{(1 - f_w(x_i))} \right\}
```

## 5. Optimization by Gradient Descent

- gradient descent step is given as follows:
```math
\begin{align}
w^{(t+1)} & \coloneqq w^{(t)} - \eta \nabla \mathcal{L}(w)\\
& \coloneqq w^{(t)} - \eta \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(w)\\
\end{align}
```
where $`\eta`$ denotes the learning rate 

## 6. Gradient

- gradient of the loss $`\mathcal{L}(w)`$ with respect to the weight $`w`$ is defined by:
```math
\nabla \mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(w),
```
where the gradient of $`\ell_i(w)`$ for each pair of input $`(x_i, y_i)`$ is defined by:
```math
\begin{align}
\nabla \ell_i(w) & = - y_i \frac{1}{f_w(x_i)} \frac{\partial f_w(x_i)}{\partial w} - (1 - y_i) \frac{1}{1 - f_w(x_i)} \frac{\partial (1 - f_w(x_i))}{\partial w}\\
& = \left( f_w(x_i) - y_i \right) x_i
\end{align}
```
where we have:
```math
\frac{d \, \sigma(z)}{d \, z} = \sigma(z) (1 - \sigma(z))
```

## 7. Initialization

- initialize all the weights $`w`$ in the neural network $`f_w`$ with $`0.0`$

## 8. Hyper-parameters

- use $`0.01`$ for the learning rate $`\eta`$
- use $`500`$ for the number of gradient descent iterations

## 9. GitHub history

- `commit` should be made at least 10 times
- the message for each `commit` should be informative with respect to the modification of the code
- the GitHub history should effectively indicate the pregress of coding

---

# [Submission]

1. [`ipynb`] jupyter notebook file (ipynb) for the complete code (filename should be `01-whatever-you-like.ipynb`)
2. [`pdf`] PDF file exported from the complete jupyter notebook file (filename should be `02-whatever-you-like.pdf`)
3. [`pdf`] PDF file for the GitHub history of the jupyter notebook file (filename should be `03-whatever-you-like.pdf`)
