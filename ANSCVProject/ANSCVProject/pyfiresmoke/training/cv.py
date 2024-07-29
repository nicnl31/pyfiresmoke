"""
This script implements cross-validation for the training object. It essentially
splits the data into k folds, and performs training on each of the folds
separately. It then averages the result, and reports the result of the best
fold.

It interacts with the Trainer object via calling it to perform training on an
individual split. It then takes the split results from the Trainer object and
stores them for comparison across splits.
"""
import os
import time
import uuid
import json
from typing import Type, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, \
	MaxAbsScaler

from base.base import BaseValidator, BaseSaver
from training.training import NeuralNetworkTrainer
from training.preprocessing import NeuralNetworkPreprocessor
from plotting.plotting import NeuralNetworkPlotter
from utils.utils import list_argmax, list_argmin, NumpyEncoder


class NeuralNetworkCVSaver(BaseSaver):
	def __init__(self, save_path):
		"""
		Initiates the Saver with a single attribute, self.checkpoints_path.
		:param path:
		"""
		# if not os.path.exists(checkpoints_path):
		# 	os.makedirs(checkpoints_path)
		# 	self.checkpoints_path = checkpoints_path
		# else:
		i = 1
		while True:
			if not os.path.exists(f"{save_path}{i}"):
				os.makedirs(f"{save_path}{i}")
				self.save_path = f"{save_path}{i}"
				break
			else:
				i += 1


class NeuralNetworkCrossValidator(BaseValidator):
	"""
	The NN cross validator stores cross-validation results for all models and
	all splits. The main attribute of the object, cv_results_, for k models with
	n splits each, has the following structure:

	self.cv_results_ = {

		'model1_name': {
			'model_kwargs': {...},
			'split_results': {
				'split1': {
					'train_scores': [...],
					'val_scores': [...]
				},
				...
				'splitn': {
					'train_scores': [...],
					'val_scores': [...]
				}
			},
			'best_split_results': {
				'best_train_scores': [train_score1, ..., train_scoren],
				'best_val_scores': [val_score1, ..., val_scoren]
			},
			'avg_results': {
				'avg_train_score': [train_score1, ..., train_scoren] / n,
				'avg_val_score': [val_score1, ..., val_scoren] / n
			},
			'optimal_n_epochs': optimal_n_epochs
		},
		...
		...
		'modelk_name': {
			'model_kwargs': {...},
			'split_results': {
				'split1': {
					'train_scores': [...],
					'val_scores': [...]
				},
				...
				'splitn': {
					'train_scores': [...],
					'val_scores': [...]
				}
			},
			'best_split_results': {
				'train_scores': [train_score1, ..., train_scoren],
				'val_scores': [val_score1, ..., val_scoren]
			},
			'avg_results': {
				'train_scores': [train_score1, ..., train_scoren] / n,
				'val_scores': [val_score1, ..., val_scoren] / n
			},
			'optimal_n_epochs': optimal_n_epochs
		}
	}
	"""
	def __init__(
			self,
			n_splits: int,
			shuffle: bool,
			scoring_method: str = 'min',
			random_state: int = 42,
			cv_path: str = "./cv_checkpoints",
			verbose: int or None = 1
	):
		assert scoring_method in ['min', 'max'], (f"scoring_method must be one "
												  f"of ['min', 'max']. It is "
												  f"currently set to be "
												  f"{scoring_method}.")
		super(NeuralNetworkCrossValidator, self).__init__(
			n_splits=n_splits,
			shuffle=shuffle,
			random_state=random_state
		)
		self.cv_saver: NeuralNetworkCVSaver = NeuralNetworkCVSaver(
			save_path=cv_path
		)
		self.cv_path: str = self.cv_saver.save_path
		self.scoring_method = scoring_method
		self.cv_results_: dict = {}
		self.verbose: int or None = verbose

		self.lr = None
		self.batch_size = None

	def cross_validate(
			self,
			model: Type[nn.Module],
			model_kwargs: dict,
			device: str,
			lr: float,
			batch_size: int,
			n_epochs: int,
			checkpoints_path: str or None,
			df_train: pd.DataFrame,
			scaler: StandardScaler or RobustScaler or MinMaxScaler or MaxAbsScaler,
			tensor_dtype: torch.dtype = torch.float32
	) -> None:
		"""
		Cross-validate a given model according to specific model settings. Then
		write results to the cv_results_ attribute.
		:param model:
		:param model_kwargs:
		:param device:
		:param lr:
		:param batch_size:
		:param n_epochs:
		:param checkpoints_path:
		:param df_train:
		:param scaler:
		:param tensor_dtype:
		:return:
		"""
		# Declare training params for later use
		self.lr = lr
		self.batch_size = batch_size

		# Initiate Stratified K-Fold object
		skf = StratifiedKFold(
			n_splits=self.n_splits,
			shuffle=self.shuffle,
			random_state=self.random_state
		)

		# Initiate Preprocessor object
		nn_preprocessor = NeuralNetworkPreprocessor()
		X_train, y_train = nn_preprocessor.get_data_and_labels(df_train, name='train')

		# Start the timer
		cv_start_time = time.time()

		# Initiate a split results dict
		split_results = {}

		for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
			# 1.1. TRAIN SPLIT: X_train and y_train (from the train indices)
			X_train_split = X_train.iloc[train_index]
			y_train_split = y_train.iloc[train_index]
			# 1.2. VAL SPLIT: X_val and y_val (from the test indices)
			X_val_split = X_train.iloc[val_index]
			y_val_split = y_train.iloc[val_index]

			X_train_scaled, X_val_scaled = nn_preprocessor.scale(
				scaler=scaler,
				X_train=X_train_split,
				X_test=X_val_split
			)

			dataloader_train, dataloader_val = nn_preprocessor.create_dataloaders(
				X_train_scaled=X_train_scaled,
				y_train=y_train_split,
				X_val_scaled=X_val_scaled,
				y_val=y_val_split,
				batch_size=batch_size,
				tensor_dtype=tensor_dtype
			)

			# 2.   NN TRAINER: create a NN trainer object and start training
			torch.manual_seed(self.random_state)
			model_split = model(**model_kwargs)
			nn_trainer = NeuralNetworkTrainer(
				model=model_split,
				device=device,
				lr=lr,
				batch_size=batch_size,
				n_epochs=n_epochs,
				save_models=False,  # Do not save models checkpoints during CV
				# checkpoints_path=f"{self.cv_path}/{checkpoints_path}",
				checkpoints_path=None,
				random_state=self.random_state
			)

			# 3.  PLOTTER: for each split
			# nn_plotter = NeuralNetworkPlotter(
			# 	model=model_split,
			# 	save_path=f"{self.cv_path}/{checkpoints_path}"
			# )

			if self.verbose and self.verbose > 0:
				print(f"\nTraining and validating for split {i+1}...")

			# 3.   TRAIN-EVAL: From the dataloaders for this split
			nn_trainer.train(train_data=dataloader_train, val_data=dataloader_val)

			# 4.   STORE RESULTS: Store split results in the attributes. Split
			#      numbers are one-indexed.
			split_results[f"split{i+1}"] = {
				"train_scores": nn_trainer.train_loss_list,
				"val_scores": nn_trainer.val_loss_list,
				"optimal_n_epochs": nn_trainer.min_val_loss_epoch + 1
			}
			# nn_plotter.plot_learning_curve(
			# 	train_loss=nn_trainer.train_loss_list,
			# 	val_loss=nn_trainer.val_loss_list,
			# 	folder_number=nn_trainer.model_saver.folder_number
			# )

		self._write_cv_results(
			model_id=str(uuid.uuid4()),
			model_kwargs=model_kwargs,
			split_results=split_results
		)
		# PRINT RESULTS
		cv_end_time = time.time()
		time_elapsed = cv_end_time - cv_start_time
		if self.verbose and self.verbose > 0:
			print(f"\n{self.n_splits}-fold cross validation finished in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.")

		return

	def get_best_model_from_cv(self, return_results=False) -> Tuple[str, dict, int] or None:
		"""
		Gets the best model from cross-validation of different models.

		:return:
			The model's:
			- Name as an UUID
			- Dictionary of kwargs
			- Optimal number of epochs
		"""
		assert len(self.cv_results_) > 0, ("Empty results. Please run "
										   "self.cross_validate() first to "
										   "generate cross-validation results.")
		models, avg_results = [], []

		for k, v in self.cv_results_.items():
			models.append(k)
			avg_results.append(v['avg_results']['avg_val_score'])

		if self.scoring_method == "min":
			best_model_idx = list_argmin(avg_results)
		elif self.scoring_method == "max":
			best_model_idx = list_argmax(avg_results)
		else:
			best_model_idx = 0

		best_model_id = models[best_model_idx]
		best_model_kwargs = self.cv_results_[best_model_id]['model_kwargs']
		best_model_optimal_n_epochs = self.cv_results_[best_model_id]['optimal_n_epochs']

		self._save_cv_results(
			save_path=self.cv_path,
			best_model_id=best_model_id,
			best_model_kwargs=best_model_kwargs,
			best_model_optimal_n_epochs=best_model_optimal_n_epochs
		)

		if self.verbose and self.verbose > 0:
			print("\nBest model from cross-validation:\n"
				  f"Model key (ID): {best_model_id}\n"
				  f"Model parameters: {best_model_kwargs}\n"
				  f"Optimal number of epochs: {best_model_optimal_n_epochs}\n\n"
				  f"Full cross validation results saved to {self.cv_path}/cv_results.json.")

		if return_results:
			return best_model_id, best_model_kwargs, best_model_optimal_n_epochs
		else:
			return

	def _save_cv_results(
			self,
			save_path,
			best_model_id,
			best_model_kwargs,
			best_model_optimal_n_epochs
	):
		"""
		Save the cross-validation results to a json file
		:return:
		"""
		with open(f"{save_path}/full_cv_results.json", "w", encoding="utf-8") as f:
			json.dump(self.cv_results_, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

		best_configs = {
			"model_id": best_model_id,
			"kwargs": best_model_kwargs,
			"optimal_n_epochs": best_model_optimal_n_epochs,
			"train_params": {
				"lr": self.lr,
				"batch_size": self.batch_size,
				"random_state": self.random_state
			}
		}
		with open(f"{save_path}/best_config.json", "w", encoding="utf-8") as g:
			json.dump(best_configs, g, ensure_ascii=False, indent=4, cls=NumpyEncoder)

	def _write_cv_results(
			self,
			model_id: str,
			model_kwargs: dict,
			split_results: dict
	) -> None:
		best_train_scores_list, best_val_scores_list, optimal_n_epochs = self._calculate_best(split_results)
		avg_train_score = sum(best_train_scores_list) / len(best_train_scores_list)
		avg_val_score = sum(best_val_scores_list) / len(best_val_scores_list)

		self.cv_results_[model_id] = {
			'model_kwargs': model_kwargs,
			'split_results': split_results,
			'best_split_results': {
				'best_train_scores': best_train_scores_list,
				'best_val_scores': best_val_scores_list,
			},
			'avg_results': {
				'avg_train_score': avg_train_score,
				'avg_val_score': avg_val_score
			},
			'optimal_n_epochs': optimal_n_epochs
		}

		return

	def _calculate_best(self, split_results):
		"""
		Helper function to calculate split results.
		:param split_results:
		:return: best_train_scores_list, best_val_scores_list, optimal_n_epochs
			The list of best train scores, best validation scores, and the
			optimal number of epochs from the best split.
		"""
		if self.scoring_method == 'min':
			best_train_scores_list = [
				min(split_results[f'split{i}']['train_scores']) for i in
				range(1, self.n_splits + 1)
			]
			best_val_scores_list = [
				min(split_results[f'split{i}']['val_scores']) for i in
				range(1, self.n_splits + 1)
			]
			best_split = list_argmin(best_val_scores_list) + 1
		elif self.scoring_method == 'max':
			best_train_scores_list = [
				max(split_results[f'split{i}']['train_scores']) for i in
				range(1, self.n_splits + 1)
			]
			best_val_scores_list = [
				max(split_results[f'split{i}']['val_scores']) for i in
				range(1, self.n_splits + 1)
			]
			best_split = list_argmax(best_val_scores_list) + 1
		else:
			best_train_scores_list, best_val_scores_list, best_split = None, None, None

		optimal_n_epochs = split_results[f'split{best_split}']['optimal_n_epochs']

		return best_train_scores_list, best_val_scores_list, optimal_n_epochs
