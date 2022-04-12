import torch


class MyNet(torch.nn.Module):
	def __init__(self, input_dim, args):
		""" MyNet is a fully convolution neural network for image semantic segmentation based on the implementation of
			Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation.
			IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.
	    Args:
	        input_dim: Amount of channels of the input
	        args: dictionary containing the arguments specified at config.py file
	    Returns:
	        torch.nn.Module: MyNet
	    .. MyNet:
	        https://github.com/kanezaki/pytorch-unsupervised-segmentation
	    """
		self.nChannel = args.COMMON.NCHANNEL
		self.nConv = args.COMMON.NCONV
		super(MyNet, self).__init__()
		self.conv1 = torch.nn.Conv2d(input_dim, self.nChannel, kernel_size=3, stride=1, padding=1)
		self.bn1 = torch.nn.BatchNorm2d(self.nChannel)
		self.conv2 = torch.nn.ModuleList()
		self.bn2 = torch.nn.ModuleList()
		for i in range(self.nConv - 1):
			self.conv2.append(torch.nn.Conv2d(self.nChannel, self.nChannel, kernel_size=3, stride=1, padding=1))
			self.bn2.append(torch.nn.BatchNorm2d(self.nChannel))
		self.conv3 = torch.nn.Conv2d(self.nChannel, self.nChannel, kernel_size=1, stride=1, padding=0)
		self.bn3 = torch.nn.BatchNorm2d(self.nChannel)
		
		self.relu = torch.nn.ReLU()
		
		self.initialize_weights()
	
	def forward(self, x):
		""" Sequentially pass `x` trough model`s convolutions and batch normalizations """
		x = self.conv1(x)
		x = self.relu(x)
		x = self.bn1(x)
		for i in range(self.nConv - 1):
			x = self.conv2[i](x)
			x = self.relu(x)
			x = self.bn2[i](x)
		x = self.conv3(x)
		x = self.bn3(x)
		return x

	def initialize_weights(self):
		""" Set initialization of each module """
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d):
				torch.nn.init.kaiming_uniform_(m.weight)
				if m.bias is not None:
					torch.nn.init.constant_(m.bias, 0)
			elif isinstance(m, torch.nn.BatchNorm2d):
				torch.nn.init.constant_(m.weight, 1)
				if m.bias is not None:
					torch.nn.init.constant_(m.bias, 0)

	def predict(self, x):
		""" Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
		Args:
			x: 4D torch tensor with shape (batch_size, channels, height, width)
		Return:
			prediction: 4D torch tensor with shape (batch_size, classes, height, width)
		"""
		if self.training:
			self.eval()

		with torch.no_grad():
			x = self.forward(x)

		return x
