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
  for i in xrange(num_train):
    scores = X[i].dot(W) # this is the prediction of training sample i, for each class
    scores -= np.max(scores)
    # calculate the probabilities that the sample belongs to each class
    probabilities = np.exp(scores) / np.sum(np.exp(scores))
    # loss is the log of the probability of the correct class
    loss += -np.log(probabilities[y[i]])

    probabilities[y[i]] -= 1 # calculate p-1 and later we'll put the negative back
    
    # dW is adjusted by each row being the X[i] pixel values by the probability vector
    for j in xrange(num_classes):
      dW[:,j] += X[i,:] * probabilities[j]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
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
  num_train = X.shape[0]
  scores = X.dot(W) # NxD * DxC = NxC
  scores -= np.max(scores,axis=1,keepdims=True)
  probabilities = np.exp(scores) / np.sum(np.exp(scores),axis=1,keepdims=True)
  correct_class_probabilities = probabilities[range(num_train),y]

  loss = np.sum(-np.log(correct_class_probabilities)) / num_train
  # that was supposed to summarize across classes that aren't classified correctly
  # so now we need to subtract 1 class for each case (a total of N) that are correctly classified
  loss += 0.5 * reg * np.sum(W*W) 

  probabilities[range(num_train),y] -= 1
  dW = X.T.dot(probabilities) / num_train

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

