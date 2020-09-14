import numpy as np

from layers import *
from optimizers import *


class CIFAR10_DataLoader(object):
    """
    Data loader class for CIFAR-10 Data.

    """
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indices = np.asarray(range(data.shape[0]))

    # reset the indices to be full length
    def _reset(self):
        self.indices = np.asarray(range(self.data.shape[0]))

    # Call this shuffle function after the last batch for each epoch
    def _shuffle(self):
        np.random.shuffle(self.indices)

    # Get the next batch of data
    def get_batch(self):
        if len(self.indices) < self.batch_size:
            self._reset()
            self._shuffle()
        indices_curr = self.indices[0:self.batch_size]
        data_batch = self.data[indices_curr]
        labels_batch = self.labels[indices_curr]
        self.indices = np.delete(self.indices, range(self.batch_size))
        return data_batch, labels_batch


def compute_acc(model, data, labels, num_samples=None, batch_size=100):
    """
    Compute the accuracy of given data and labels

    """
    N = data.shape[0]
    if num_samples is not None and N > num_samples:
        indices = np.random.choice(N, num_samples)
        N = num_samples
        data = data[indices]
        labels = labels[indices]

    num_batches = N // batch_size
    if N % batch_size != 0:
        num_batches += 1
    preds = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        output = model.forward(data[start:end], False)
        scores = softmax(output)
        pred = np.argmax(scores, axis=1)
        preds.append(pred)
    preds = np.hstack(preds)
    accuracy = np.mean(preds == labels)
    return accuracy


""" Some comments """
def train_net(data, model, loss_func, optimizer, batch_size, max_epochs,
              lr_decay=1.0, lr_decay_every=1000, show_every=10, verbose=False):
    """
    Train a network with this function, parameters of the network are updated
    using stochastic gradient descent methods defined in optim.py.

    The parameters which achive the best performance after training for given epochs
    will be returned as a param dict. The training history and the validation history
    is returned for post analysis.

    """

    # Initialize the variables
    data_train, labels_train = data["data_train"]
    data_val, labels_val = data["data_val"]
    dataloader = CIFAR10_DataLoader(data_train, labels_train, batch_size)
    opt_val_acc = 0.0
    opt_params = None
    loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    # Compute the maximum iterations and iterations per epoch
    iters_per_epoch = int(max(data_train.shape[0] / batch_size, 1))
    max_iters = int(iters_per_epoch  * max_epochs)

    # Start the training
    for epoch in range(max_epochs):
        # Compute the starting iteration and ending iteration for current epoch
        iter_start = epoch * iters_per_epoch
        iter_end   = (epoch + 1) * iters_per_epoch

        # Decay the learning rate every specified epochs
        if epoch % lr_decay_every == 0 and epoch > 0:
            optimizer.lr = optimizer.lr * lr_decay
            print ("Decaying learning rate of the optimizer to {}".format(optimizer.lr))

        # Main training loop
        for iter in range(iter_start, iter_end):
            data_batch, labels_batch = dataloader.get_batch()

            scores = model.forward(data_batch)
            loss = loss_func.forward(scores, labels_batch)
            loss_hist.append(loss)

            dloss = loss_func.backward()
            din = model.backward(dloss)
            optimizer.net = model.net
            optimizer.step()
            model.net = optimizer.net

            # Show the training loss
            if verbose and iter % show_every == 0:
                print ("(Iteration {} / {}) loss: {}".format(iter+1, max_iters, loss_hist[-1]))

        # End of epoch, compute the accuracies
        train_acc = 0
        val_acc = 0
        train_acc = compute_acc(model, data_train, labels_train)
        val_acc = compute_acc(model, data_val, labels_val)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        # Save the best params for the model
        if val_acc > opt_val_acc:
            opt_params = model.net.params

        # Show the training accuracies
        if verbose:
            print ("(Epoch {} / {}) Training Accuracy: {}, Validation Accuracy: {}".format(
            epoch+1, max_epochs, train_acc, val_acc))

    return opt_params, loss_hist, train_acc_hist, val_acc_hist
