from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from base import BasePlotter, BaseSaver


class NeuralNetworkPlotSaver(BaseSaver):
	def __init__(self, save_path: str):
		"""
		Initiates the Saver with a single attribute, self.checkpoints_path.
		:param save_path:
		"""
		super(NeuralNetworkPlotSaver, self).__init__(save_path=save_path)


class NeuralNetworkPlotter(BasePlotter):
	def __init__(
			self,
			model: Any,
			save_path: str
	):
		self.model = model
		self.saver = NeuralNetworkPlotSaver(save_path=save_path)

	def plot_learning_curve(self, train_loss, val_loss=None):
		"""
		Plot the learning curve, and save
		:param train_loss:
		:param val_loss:
		:return:
		"""
		plt.figure(figsize=(12,8), dpi=200)
		plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
		if val_loss:
			plt.plot(range(1, len(train_loss)+1), val_loss, label='Validation Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title(f'{self.model.__class__.__name__} learning curve', weight='bold')
		plt.legend()
		plt.savefig(f"{self.saver.save_path}/learning_curve.png")

	def plot_confusion_matrix(self, y_true, y_pred, normalize='true') -> None:
		mtx = ConfusionMatrixDisplay.from_predictions(
			y_true=y_true,
			y_pred=y_pred,
			normalize=normalize
		)
		if normalize == 'true':
			mtx.figure_.suptitle(f"Normalised confusion matrix")
			plt.savefig(f"{self.saver.save_path}/confusion_matrix_norm.png")
		if normalize is None:
			mtx.figure_.suptitle(f"Confusion matrix")
			plt.savefig(f"{self.saver.save_path}/confusion_matrix.png")
		return
