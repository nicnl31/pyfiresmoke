import torch.nn as nn
from torch.nn.modules.container import ModuleList

from mapper import StringToFunctionMapper


class ANN(nn.Module):
	"""
	Artificial Neural Network with a specified number of hidden layers and
	neurons per layer.
	"""
	def __init__(
			self,
			input_size: int,
			output_size: int,
			hidden_size: int,
			dropout_rate: int or float,
			n_hiddens: int,
			activation_function: str
	):
		super(ANN, self).__init__()
		self.fully_connected: ModuleList = nn.ModuleList()
		for h in range(n_hiddens):
			# First layer after the input layer
			if h == 0:
				self.fully_connected.append(nn.Linear(input_size, hidden_size))
			# Middle layers
			else:
				self.fully_connected.append(nn.Linear(hidden_size, hidden_size))

			activation = StringToFunctionMapper().map(
				to_map=activation_function,
				mapper_type="activation"
			)
			self.fully_connected.append(activation())
			self.fully_connected.append(nn.Dropout(dropout_rate))
		self.fully_connected.append(nn.Linear(hidden_size, output_size))

	def forward(self, x):
		for layer in self.fully_connected:
			x = layer(x)
		return x
