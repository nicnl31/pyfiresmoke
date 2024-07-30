"""
This script implements pre-processing steps for the dataset.
"""
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, \
	MaxAbsScaler
from torch.utils.data import DataLoader, TensorDataset
from torchsampler import ImbalancedDatasetSampler

from base import BasePreprocessor


class NeuralNetworkPreprocessor(BasePreprocessor):
	def __init__(self):
		super(NeuralNetworkPreprocessor, self).__init__()

	def get_data_and_labels(
			self,
			df: pd.DataFrame,
			name: str = 'train'
	) -> Tuple:
		"""
		Separates the data and labels.

		:param df: pd.DataFrame
			The DataFrame object.
		:param name:
			The name of the set. One of ['train', 'test'].
		:return:
		"""
		# Copy the datasets to avoid directly modifying the original DataFrames
		df_copy = df.copy()

		# Separate the data, label, and image names
		y = df_copy.pop('label')
		img = df_copy.pop('image')
		X = df_copy

		if name == 'train':
			self.X_train_ = X
			self.y_train_ = y
			self.img_train_ = img
		elif name == 'test':
			self.X_test_ = X
			self.y_test_ = y
			self.img_test_ = img
		elif name not in ['train', 'test']:
			warnings.warn(f'Invalid name "{name}" for the data. The name should be one of "train" or "test".')

		return X, y

	def scale(
			self,
			scaler: StandardScaler or RobustScaler or MinMaxScaler or MaxAbsScaler,
			X_train: pd.DataFrame,
			X_test: pd.DataFrame
	) -> Tuple[np.ndarray, np.ndarray]:
		s = scaler()
		X_train_scaled = s.fit_transform(X_train)
		X_test_scaled = s.transform(X_test)
		self.scaler_ = s

		return X_train_scaled, X_test_scaled

	def create_dataloaders(
			self,
			X_train_scaled: np.ndarray,
			y_train: pd.Series,
			X_val_scaled: np.ndarray,
			y_val: pd.Series,
			batch_size: int,
			tensor_dtype: torch.dtype = torch.float32,
	) -> Tuple[DataLoader, DataLoader]:
		"""
		Creates PyTorch DataLoader objects for the neural network trainer.
		Assumes the training data has been split into a sub-training set and a
		validation set.

		:param X_train_scaled: np.ndarray
			The training data, already scaled.
		:param y_train: pd.Series
			The training labels.
		:param X_val_scaled: np.ndarray
			The validation data, already scaled.
		:param y_val: pd.Series
			The validation labels.
		:param batch_size: int
			The batch size required for the DataLoader object.
		:param tensor_dtype:
			The tensor dtype conversion for the TensorDataset object.
		:return:
			(dataloader_train, dataloader_val)
			The train and validation DataLoader objects.
		"""
		# TENSORISE: convert to tensors
		X_train_scaled_tsr = torch.tensor(X_train_scaled, dtype=tensor_dtype)
		X_val_scaled_tsr = torch.tensor(X_val_scaled, dtype=tensor_dtype)
		y_train_tsr = torch.tensor(y_train.values, dtype=tensor_dtype)
		y_val_tsr = torch.tensor(y_val.values, dtype=tensor_dtype)

		# TENSORDATASET: package the data into TensorDataset objects
		dataset_train = TensorDataset(X_train_scaled_tsr, y_train_tsr)
		dataset_val = TensorDataset(X_val_scaled_tsr, y_val_tsr)

		# DATALOADER: Create dataloader objects for this split
		dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
									  sampler=ImbalancedDatasetSampler(dataset_train))
		dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

		return (dataloader_train, dataloader_val)
