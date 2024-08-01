import os
import time
from typing import Tuple

import numpy as np
from numpy import ndarray

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from base import BaseTrainer


class NeuralNetworkModelSaver(object):
	"""
	Saves model weights every epoch as long as the validation loss decreases.

	As a new version is saved, all old versions are deleted, thus saves storage.

	"""
	def __init__(self, model, checkpoints_path="./training_checkpoints"):
		self.model = model
		self.min_val_loss = np.inf
		self.epochs = []

		if not os.path.exists(checkpoints_path):
			os.makedirs(checkpoints_path)
		self.checkpoints_path = checkpoints_path

	def save(self, epoch_num, val_loss):
		# Save model weights if validation loss decreases
		if val_loss < self.min_val_loss:  # Keeps track of the minimum validation loss
			self.min_val_loss = val_loss
			self.epochs.append(epoch_num)
			torch.save(
				self.model.state_dict(),
				f'{self.checkpoints_path}/{self.model.__class__.__name__}_ep{epoch_num}.pt'
			)
			if len(self.epochs) >= 2:
				os.remove(f'{self.checkpoints_path}/{self.model.__class__.__name__}_ep{self.epochs[-2]}.pt')

	def get_best_model_path(self):
		"""
		Retrieves the path of the best model when training finishes.
		"""
		return f'{self.checkpoints_path}/{self.model.__class__.__name__}_ep{self.epochs[-1]}.pt'


class NeuralNetworkEarlyStoppingValidator(object):
	"""
	A custom validator class for early stopping conditions. It sets the threshold for
	early stopping of network training based on the divergence between training loss
	and validation loss.

	If validation loss is greater than the current minimum validation loss plus some
	threshold for a period of time exceeding the "patience" level, the training loop
	should break.
	"""

	def __init__(self, patience, delta):
		self.patience = patience
		self.delta = delta
		self.min_val_loss = np.inf
		self.counter = 0

	def early_stop(self, val_loss):
		"""
		Validate the early stopping conditions based on the observed validation loss.
		"""
		if val_loss < self.min_val_loss:  # Keeps track of the minimum validation loss
			self.min_val_loss = val_loss
			self.counter = 0  # reset the counter
		elif val_loss >= self.min_val_loss + self.delta:
			self.counter += 1
			if self.counter >= self.patience:
				return True
		return False


class NeuralNetworkTrainer(BaseTrainer):
	"""
	A child class of the BaseTrainer that will input data and train the neural
	network model on specified settings. It also stores all datasets and
	dataloaders.
	"""

	def __init__(
			self,
			model: nn.Module,
			device: str,
			lr: float,
			batch_size: int,
			n_epochs: int,
			random_state: int,
			save_models: bool = True,
			checkpoints_path: str = None,
			verbose: int or None = 1
	):
		# FOR PRINTING OUTPUTS
		self.verbose = verbose

		# TRAINING PARAMETERS
		self.model: nn.Module = model
		self.device: str = device
		self.lr: float = lr
		self.batch_size: int = batch_size
		self.n_epochs: int = n_epochs
		self.random_state: int = random_state

		# CALLBACKS: Loss function, optimizer, learning rate scheduler, early stopper
		self.loss_fn = self._get_loss_fn()
		self.optimizer = self._get_optimizer()
		self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
		self.early_stopper = self._get_early_stopper()

		# SAVING: Model saver for weight saving
		self.save_models = save_models
		if self.save_models:
			if not checkpoints_path:
				self.model_saver = self._get_model_saver(
					self.model,
					checkpoints_path="./training_checkpoints"
				)
			elif checkpoints_path:
				self.model_saver = self._get_model_saver(
					self.model,
					checkpoints_path=checkpoints_path
				)
		else:
			self.model_saver = None

		# STORING: Model training results for storing
		self.train_loss_list = []
		self.val_loss_list = []

		# STORING: Best validation epoch and best model
		self.min_train_loss_epoch = None
		self.min_val_loss_epoch = None  # Zero-based epoch number

	def _get_loss_fn(self, weight=None):
		return nn.CrossEntropyLoss(weight=weight)

	def _get_optimizer(self):
		"""
		Statically sets the Adam optimizer for the trainer.
		"""
		return optim.Adam(self.model.parameters(), lr=self.lr)

	def _get_lr_scheduler(self, optimizer):
		"""
		Sets the ReduceLROnPlateau with the specified optimizer. The scheduler
		will reduce the learning rate
		"""
		return ReduceLROnPlateau(optimizer, mode='min', threshold=1e-4,
								 threshold_mode='rel', factor=1e-1, patience=10,
								 min_lr=1e-8)

	def _get_early_stopper(self, patience=10, delta=1e-6):
		return NeuralNetworkEarlyStoppingValidator(patience, delta)

	def _get_model_saver(self, model, checkpoints_path):
		return NeuralNetworkModelSaver(model, checkpoints_path=checkpoints_path)

	def train(
			self,
			train_data: DataLoader,
			val_data: DataLoader
	) -> None:
		"""
		Train and validate the model.
		:param train_data: DataLoader
			The training DataLoader object.
		:param val_data: DataLoader
			The validation DataLoader object.
		:return:
		"""
		# Send the model to the specified device
		self.model.to(self.device)
		# Start the timer
		training_start_time = time.time()
		# Loop through the epochs
		for epoch in range(self.n_epochs):
			# ==================================================================
			# 1. EPOCH INITIALISATION: Training and validation loss
			epoch_train_loss, epoch_val_loss = 0.0, 0.0

			# ==================================================================
			# 2. EPOCH TRAINING LOOP
			self.model.train()
			for train_X, train_y in train_data:
				# Move data and target to the same device as the model, and
				# convert the labels to torch.long to match expected output
				train_X_device = train_X.to(self.device)
				train_y_device = train_y.to(self.device).to(torch.long)
				# Set zero gradients
				self.optimizer.zero_grad()
				# Forward Propagation to get predicted outcome
				train_pred_y_device = self.model(train_X_device)
				# Train loss
				train_loss = self.loss_fn(train_pred_y_device, train_y_device)
				# Back propagation
				train_loss.backward()
				# Update the weights
				self.optimizer.step()
				# Add the batch loss and f1_score to the running total
				epoch_train_loss += train_loss.item()

			# ==================================================================
			# EPOCH VALIDATION LOOP
			self.model.eval()
			with torch.no_grad():
				for val_X, val_y in val_data:
					# Move data and target to the same device as the model
					val_X_device = val_X.to(self.device)
					val_y_device = val_y.to(self.device).to(torch.long)
					# Propagate forward to get validation predictions
					val_pred_y_device = self.model(val_X_device)
					# Compute the validation loss
					val_loss = self.loss_fn(val_pred_y_device, val_y_device)
					# Add the batch loss to the running epoch loss total
					epoch_val_loss += val_loss.item()

			# ==================================================================
			# POST-EPOCH CALCULATIONS: TRAIN-VALIDATION LOSS
			epoch_train_loss /= len(train_data)
			epoch_val_loss /= len(val_data)

			# ==================================================================
			# Update learning rate scheduler with epoch validation loss
			self.lr_scheduler.step(epoch_val_loss)

			# Save model if validation loss decreases
			if self.save_models:
				self.model_saver.save(epoch_num=epoch, val_loss=epoch_val_loss)

			# Store epoch results
			self.train_loss_list.append(epoch_train_loss)
			self.val_loss_list.append(epoch_val_loss)

			# ==================================================================
			# PRINT RESULTS AND EVALUATE EARLY STOPPING
			if self.verbose and self.verbose > 0:
				if (epoch + 1 == 1) or ((epoch + 1) % 10 == 0):
					epoch_end_time = time.time()
					epoch_elapsed = epoch_end_time - training_start_time
					print(
						f"EPOCH [{epoch + 1}/{self.n_epochs}],\t Train Loss: {epoch_train_loss:.3f}, Val Loss: {epoch_val_loss:.3f}, Elapsed time: {epoch_elapsed // 60:.0f}m {epoch_elapsed % 60:.0f}s")

			# Early stopping
			if self.early_stopper.early_stop(epoch_val_loss):
				break

		# FINISH TRAINING: Update results
		training_end_time = time.time()
		time_elapsed = training_end_time - training_start_time
		min_val_loss_idx = np.argmin(self.val_loss_list)
		self.min_val_loss_epoch = min_val_loss_idx

		print(f'\nTraining finished in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.\n'
			  f'Total epochs trained: {len(self.val_loss_list)}\nBest epoch: {self.min_val_loss_epoch}\n'
			  f'Best validation loss: {self.val_loss_list[self.min_val_loss_epoch]}'
		)
		return

	def train_without_validation(
			self,
			train_data: DataLoader
	) -> None:
		"""
		Train the model without validating on a validation set.
		:param train_data: DataLoader
			The training DataLoader object.
		:return:
		"""
		# Send the model to the specified device
		self.model.to(self.device)
		# Start the timer
		training_start_time = time.time()
		# Loop through the epochs
		for epoch in range(self.n_epochs):
			# ==================================================================
			# 1. EPOCH INITIALISATION: Training loss
			epoch_train_loss = 0.0

			# ==================================================================
			# 2. EPOCH TRAINING LOOP
			self.model.train()
			for train_X, train_y in train_data:
				# Move data and target to the same device as the model, and
				# convert the labels to torch.long to match expected output
				train_X_device = train_X.to(self.device)
				train_y_device = train_y.to(self.device).to(torch.long)
				# Set zero gradients
				self.optimizer.zero_grad()
				# Forward Propagation to get predicted outcome
				train_pred_y_device = self.model(train_X_device)
				# Train loss
				train_loss = self.loss_fn(train_pred_y_device, train_y_device)
				# Back propagation
				train_loss.backward()
				# Update the weights
				self.optimizer.step()
				# Add the batch loss and f1_score to the running total
				epoch_train_loss += train_loss.item()

			# ==================================================================
			# POST-EPOCH CALCULATIONS: TRAIN-VALIDATION LOSS
			epoch_train_loss /= len(train_data)

			# Save model if train loss decreases
			if self.save_models:
				self.model_saver.save(epoch_num=epoch, val_loss=epoch_train_loss)

			# Store epoch results
			self.train_loss_list.append(epoch_train_loss)

			# ==================================================================
			# PRINT RESULTS AND EVALUATE EARLY STOPPING
			if self.verbose and self.verbose > 0:
				if (epoch + 1 == 1) or ((epoch + 1) % 10 == 0):
					epoch_end_time = time.time()
					epoch_elapsed = epoch_end_time - training_start_time
					print(
						f"EPOCH [{epoch + 1}/{self.n_epochs}],\t Train Loss: {epoch_train_loss:.3f}, Elapsed time: {epoch_elapsed // 60:.0f}m {epoch_elapsed % 60:.0f}s")

		# FINISH TRAINING: Update results
		training_end_time = time.time()
		time_elapsed = training_end_time - training_start_time
		min_train_loss_idx = np.argmin(self.train_loss_list)
		self.min_train_loss_epoch = min_train_loss_idx

		print(
			f'\nTraining finished in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.\n'
			f'Total epochs trained: {len(self.train_loss_list)}\nBest epoch: {self.min_train_loss_epoch}\n'
			f'Best train loss: {self.train_loss_list[self.min_train_loss_epoch]}'
		)
		return

	def test(
			self,
			test_data: DataLoader,
			best_model_path: str
	) -> np.ndarray:
		self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
		self.model.eval()
		y_true_concat, y_pred_concat = torch.LongTensor(), torch.LongTensor()
		with torch.no_grad():
			for X, y in test_data:
				X_device = X.to(self.device)
				y_device = y.to(self.device)

				y_pred_device = self.model(X_device)
				y_pred_device = torch.argmax(y_pred_device, dim=1)

				y_true_concat = torch.cat((y_true_concat, y_device))
				y_pred_concat = torch.cat((y_pred_concat, y_pred_device))

		return y_pred_concat.numpy()
