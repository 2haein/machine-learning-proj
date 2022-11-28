# Image Inpainting

## 1. Baseline notebook code

- [assignment_11.ipynb](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/11/assignment_11.ipynb)

## 2. Data

- [assignment_11_data.npz](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/11/assignment_11_data.npz)
- data consists of `training` data and `testing` data
- `training` data is used for training a neural network
- `testing` data is used for validating the trained neural network
- `training` data consists of desired images without degraded ones
- `testing` data consists of pairs of degraded images and their ground truth

## 3. Neural Network

- you can construct any neural network architecture in the form of auto-encoder 
- you can consider Sigmoid function for the activation function of the output layer
- construct a neural network architecture in such a way that best testing accuracy can be achieved

## 4. Loss

- you can use any loss

## 5. Optimization

- you can use any optimizer

## 6. Training

- you should construct your own training dataset based on the provided training data that consist of only desired images

## 7. Testing

- you should evaluate your model based on the provided testing data that consist of both degraded images and their ground truth

## 8. Initialization

- initialization of the weights of the neural network architecture in such a way that best testing accuracy can be achieved 

## 9. Evaluation

- the performance of the model is evaluated by PSNR
- PSNR is computed by $`PSNR = 10 * \log_{10} \left( \frac{MAX(IMAGE)^2}{MSE} \right)`$
- set $`MAX(IMAGE) = 1`$ for our dataset
- $`MSE(x, y) = \frac{1}{n} \| x - y \|_2^2`$ where $`n`$ is the size of data $`x`$ and $`y`$
 
## 10. Hyper-parameters

- determine the hyper-parameters in such a way that best testing accuracy can be achieved

## 11. Grading

- the scores are given by the ranking of the final testing accuracy within the correct answers
  - rank 01 - 05 : 10
  - rank 05 - 10 : 9
  - rank 11 - 15 : 8
  - rank 15 - 20 : 7
  - rank 21 - 25 : 6
  - rank 25 - 30 : 5
  - rank 31 - 35 : 4
  - rank 35 - 40 : 3
  - rank 41 - : 2

## 12. GitHub history

- `commit` should be made at least 10 times
- the message for each `commit` should be informative with respect to the modification of the code
- the GitHub history should effectively indicate the pregress of coding

## [Submission]

1. [x] jupyter notebook file (ipynb) for the complete code (filename should be `01-whatever-you-like.ipynb`)
2. [x] PDF file exported from the complete jupyter notebook file (filename should be `02-whatever-you-like.pdf`)
3. [x] PDF file for the GitHub history of the jupyter notebook file (filename should be `03-whatever-you-like.pdf`)
