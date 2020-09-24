import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
    scores = X[i].dot(W)
    scores = scores - np.max(scores)
    correct_class_score = scores[y[i]]
    denom = 0.0
   
    for j in range(num_classes):
      denom += np.exp(scores[j])
    
    exps = np.exp(correct_class_score) / denom
    loss += -np.log(exps)
    
    # Calculate dW for each class
    for j in range(num_classes):
      dW[:, j] += X[i] * ((np.exp(scores[j]) / denom) - int(j == y[i]))
    
  #We need to average so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # dW regularization gradient
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # Calculate scores for each training sample
  scores = X.dot(W)
  scores = scores - np.max(scores)
    
  # Calculate sum of exps for denom
  denom = np.sum(np.exp(scores), axis=1)

  # Create exp fraction for each correct score / sum
  exps = np.exp(scores[range(X.shape[0]), y]) / denom
  # Compute loss
  loss = np.mean(-np.log(exps))

  # Compute dW using derivative and mask to get correct scores
  mask = np.zeros((X.shape[0], W.shape[1]))
  mask[range(X.shape[0]), y] = 1 # Set correct scores to 1
  dW = X.T.dot((np.exp(scores) / denom[:, np.newaxis]) - mask) / X.shape[0]

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # dW regularization gradient
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

