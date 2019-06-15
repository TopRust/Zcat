import gstate
import time
from torch.autograd import Variable

# the encapsulation class to train model
class Trainer():
	def __init__(self, use_cuda, loader_train, max_epoch, experiment=None, updater=None, supervised=True):
		self.use_cuda = use_cuda
		self.loader_train = loader_train
		self.max_epoch = max_epoch
		self.supervised = supervised
		self.experiment = experiment
		self.updater = updater
		self.headtrainsions = []
		self.headepochsions = []
		self.tailepochsions = []
		self.tailtrainsions	= []

	def run(self):
		gstate.set_value('start_time', time.time())
		self.head_train()
		for i in range(gstate.get('epoch'), self.max_epoch + 1):
			gstate.set_value('epoch', i)
			gstate.clear_statics()
			self.experiment.train()
			self.head_epoch()

			for x, t in self.loader_train:
				# supervised task
				if not self.supervised:
					if self.use_cuda:
						x = x.cuda()
					x = x.float()
					x = Variable(x)
					self.updater.zero_grad()
					loss = self.experiment(x)
				# unsupervised task
				else:
					if self.use_cuda:
						x, t = x.cuda(), t.cuda()
					x = x.float()
					x, t = Variable(x), Variable(t)
					self.updater.zero_grad()
					loss = self.experiment(x, t)
								
				loss.backward()
				self.updater.step()

			self.tail_epoch()
		self.tail_train()

	def headtrain(self, function, * args):
		self.headtrainsions.append((function, args))

	def headepoch(self, function, * args):
		self.headepochsions.append((function, args))

	def tailepoch(self, function, *args):
		self.tailepochsions.append((function, args))

	def tailtrain(self, function, *args):
		self.tailtrainsions.append((function, args))


	def head_train(self):
		for f, args in self.headtrainsions:
			args = (self, ) + args
			f(*args)

	def head_epoch(self):
		for f, args in self.headepochsions:
			args = (self, ) + args
			f(*args)

	def tail_epoch(self):
		for f, args in self.tailepochsions:
			args = (self, ) + args
			f(*args)

	def tail_train(self):
		for f, args in self.tailtrainsions:
			args = (self, ) + args
			f(*args)

# import gstate
# import dataset
# import model
# import updater
# import experiment
# import extensions
# import torch
# import torch.optim as optim

# if __name__ == '__main__':

# 	trainloader, testloader = dataset.cifar10_train_loader()
# 	testloader = dataset.cifar10_test_loader()
# 	predictor = model.ResNet18()
# 	net = experiment.E_basic(predictor)
# 	learning_rate = 0.1
# 	decay = 1e-4
# 	updater = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)
# 	max_epoch = 400
# 	net = experiment.E_basic(predictor)

# 	use_cuda = torch.cuda.is_available()
# 	trainer = Trainer(use_cuda, trainloader, net, updater, max_epoch)
# 	trainer.tailepoch(extensions.test, trainer, testloader, use_cuda)
# 	train.tailepoch(extensions.)
# 	trainer.run()
