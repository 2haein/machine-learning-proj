# Classification for multiple classes using Pytorch

- pytorch
- neural network architecture
- stochastic gradient descent
- weight decay
- softmax and cross entropy
- training and testing

## 0. Tutorial on Pytorch

- [PyTorch tutorials](https://pytorch.org/tutorials/)

## 1. Baseline notebook code

- [assignment_06.ipynb](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/06/assignment_06.ipynb)

## 2. Data

- [assignment_06_data.npz](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/06/assignment_06_data.npz)
- data consists of `training` data and `testing` data
- `training` data is used for training a neural network
- `testing` data is used for validating the trained neural network
- each data consist of pairs of images and their class labels
- images represent digits from 0 to 9 and labels represent their classes

## 3. Neural Network

- construct a neural network using pytorch library
- use 2-dimensional convolution `Conv2d`, 2-dimensional maxpooling `MaxPool2d`, and ReLU `ReLU` for the feature layers
- use fully connected layer `Linear` and ReLU `ReLU` for the classification layers
- the final layer should be fully connected layer `Linear` so that the loss function `CrossEntropyLoss` can be used (do not add `Softmax` at the output layer)

## 4. Loss

- use the softmax and the cross-entropy
- use `CrossEntropyLoss` in the pytorch library
- use a fully connected layer `Linear` for the output of the neural network since the loss function `CrossEntropyLoss` combines the softmax and the cross-entropy loss

## 5. Optimization

- use stochastic gradient descent (sgd) optimizer in the pytorch library
- use a constant learning rate (do not use learning rate annealing scheme such as Adam, AdaGrad, RMSProp)
- you can use weight decay

## 6. Training

- training aims to determine the model parameters of the neural network and its associated loss function is minimized using the training data

## 7. Testing

- testing aims to validate the generality of the trained neural network using the testing data

## 8. Initialization

- initialize all the weights in the neural network in such a way that best testing accuracy can be achieved
  
## 9. Hyper-parameters

- determine the followings in such a way that best testing accuracy can be achieved
  - neural network architecture
  - initialization of the weights
  - number of epochs
  - size of mini-batch
  - learning-rate
  - weight-decay

## 10. GitHub history

- `commit` should be made at least 10 times
- the message for each `commit` should be informative with respect to the modification of the code
- the GitHub history should effectively indicate the pregress of coding

## 11. Grading

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
