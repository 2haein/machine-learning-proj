# Image Generation via Generative Adversarial Networks

## 1. Baseline notebook code

- [assignment_12.ipynb](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/12/assignment_12.ipynb)

## 2. Data

- [assignment_12_data.npz](https://gitlab.com/cau-class/neural-network/2022-2/assignment/-/blob/master/12/assignment_12_data.npz)
- the provided dataset consists of only training data that are to be used for training a generative model
- `training` data consists of binary images for squre shapes in varying sizes and locations

## 3. Neural Network

- you can use any neural network architectures for the discriminator and the generator in such a way that best testing accuracy can be achieved

## 4. Loss

- you can use any loss

## 5. Optimization

- you can use any optimizer

## 6. Training

- you should use the training data for training your model

## 7. Testing

- you should use latent vectors for generating fake images at the end of your training

## 8. Initialization

- initialization of the weights of the neural networks in such a way that best testing accuracy can be achieved 

## 9. Evaluation

- the performance of the trained model is evaluated by the intersection over union (IoU) between the generated images and their bounding box images
```math
\textrm{IoU} = \frac{\textrm{Area of Intersection}}{\textrm{Area of Union}}
```
- the thresholding with a threshold 0.5 is applied to the generated images for the evaluation
 
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






