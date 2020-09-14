import numpy as np
form datasets import CIFAR10_data
from layers import *
from optimizers import *
from train import *


class SimpleNet(Module):
    def __init__(self, keep_prob=0.5, dtype=np.float32, seed=None):
        """ Some comments """
        self.dropout = dropout
        self.seed = seed
        self.net = sequential(
            flatten(name="flat"),
            fc(3072, 500, 1e-2, name="fc1"),
            dropout(keep_prob, seed=seed),
            relu(name="relu1"),
            fc(500, 500, 1e-2, name="fc2"),
            relu(name="relu2"),
            fc(500, 10, 1e-2, name="fc3"),
        )

def main():
	"""Load data"""
	data = CIFAR10_data()

	"""Build model and set hyper-parameters"""
	model = SimpleNet()
	loss_f = cross_entropy()
	optimizer = SGD(model.net, 1e-4)
	batch_size = 16
	epochs = 10
	lr_decay = 0.99

	"""Train"""
	results = train_net(data_dict, model, loss_f, optimizer, batch_size, epochs, 
	                    lr_decay, show_every=10000, verbose=True)
	opt_params, loss_hist, train_acc_hist, val_acc_hist = results


	"""Display result"""
	al_acc = compute_acc(model, data["data_val"], data["labels_val"])
	print ("Validation Accuracy: {}%".format(val_acc*100))
	test_acc = compute_acc(model, data["data_test"], data["labels_test"])
	print ("Testing Accuracy: {}%".format(test_acc*100))	


if __name__ == '__main__':
	main()
