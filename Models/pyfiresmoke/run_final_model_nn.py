"""
This script runs the final training loop to get the best model based on
cross-validation results.

It also tests the trained model on the final test set, and stores test results
and confusion matrix.

"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

from training.preprocessing import NeuralNetworkPreprocessor
from training.training import NeuralNetworkTrainer
from plotting import NeuralNetworkPlotter
from utils import parse, NumpyEncoder

# ==============================================================================
# MODEL IMPORTS GO HERE
from models.ann import ANN

# MODEL HYPERPARAMETER SETTINGS: Specify model save path and kwargs path
SAVE_PATH = "../../results/final_models/final_model_hsv_60features1"  # <-- Change this as needed 
MODEL_CONFIG_PATH = "../../results/cv/cv_checkpoints_hsv_60features1"  # <-- Change this as needed 
with open(f"{MODEL_CONFIG_PATH}/best_config.json", mode='r') as f:
	BEST_CONFIG = json.load(f)

MODEL_KWARGS = BEST_CONFIG["kwargs"]
OPTIMAL_N_EPOCHS = BEST_CONFIG["optimal_n_epochs"]
LR = BEST_CONFIG["train_params"]["lr"]
BATCH_SIZE = BEST_CONFIG["train_params"]["batch_size"]
RANDOM_STATE = BEST_CONFIG["train_params"]["random_state"]

# CHANGE MODEL HERE
torch.manual_seed(RANDOM_STATE)
MODEL = ANN(**MODEL_KWARGS)

# DATASET VARIABLES: CHANGE AS NEEDED
dataset_dir = '../../data'  # <-- Change this as needed
dataset_name = 'fasdd_dataset'  # <-- Change this as needed
df_name = 'csv/dataset_fasdd_no_lbp_HSV_60features.csv'  # <-- Change this as needed
split_dir = 'annotations/YOLO_CV'
train_filename = 'train.txt'
val_filename = 'val.txt'
test_filename = 'test.txt'


# ==============================================================================


def main() -> None:
	df = pd.read_csv(f"{dataset_dir}/{dataset_name}/{df_name}")

	train_split = parse(dataset_dir, dataset_name, split_dir, train_filename)
	val_split = parse(dataset_dir, dataset_name, split_dir, val_filename)
	test_split = parse(dataset_dir, dataset_name, split_dir, test_filename)

	df_train = df[df["image"].isin(train_split)]
	df_val = df[df["image"].isin(val_split)]
	df_test = df[df["image"].isin(test_split)]

	df_train = pd.concat([df_train, df_val])

	# PREPROCESSOR OBJECT
	preprocessor = NeuralNetworkPreprocessor()

	# DATA/LABEL for train/test
	preprocessor.get_data_and_labels(
		df=df_train,
		name='train'
	)
	preprocessor.get_data_and_labels(
		df=df_test,
		name='test'
	)
	X_train = preprocessor.X_train_
	X_test = preprocessor.X_test_
	y_train = preprocessor.y_train_
	y_test = preprocessor.y_test_

	# SCALE
	X_train_scaled, X_test_scaled = preprocessor.scale(X_train=X_train,
													   X_test=X_test,
													   scaler=StandardScaler)

	# DATALOADERS
	dataloader_train, dataloader_test = preprocessor.create_dataloaders(
		X_train_scaled=X_train_scaled,
		X_val_scaled=X_test_scaled,
		y_train=y_train,
		y_val=y_test,
		batch_size=BATCH_SIZE
	)

	# TRAINER
	trainer = NeuralNetworkTrainer(
		model=MODEL,
		device="cpu",
		lr=LR,
		batch_size=BATCH_SIZE,
		n_epochs=OPTIMAL_N_EPOCHS,
		random_state=RANDOM_STATE,
		save_models=True,
		checkpoints_path=SAVE_PATH
	)

	# PLOTTER
	plotter = NeuralNetworkPlotter(model=MODEL, save_path=SAVE_PATH)

	# EXECUTION
	trainer.train_without_validation(train_data=dataloader_train)
	y_pred = trainer.test(
		test_data=dataloader_test,
		best_model_path=trainer.model_saver.get_best_model_path()
	)

	# METRICS
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred, average="macro")
	recall = recall_score(y_test, y_pred, average="macro")
	clf_stats = f"acc:{accuracy:.2f}, prec: {precision:.2f}, rec: {recall:.2f}" 
	
	# PLOT
	plotter.plot_learning_curve(train_loss=trainer.train_loss_list)
	plotter.plot_confusion_matrix(y_true=y_test, y_pred=y_pred,
								  normalize='true', extra_stats=clf_stats)
	plotter.plot_confusion_matrix(y_true=y_test, y_pred=y_pred,
								  normalize=None, extra_stats=clf_stats)
	
	print(f"\nAccuracy: {accuracy:.2f}\n"
		  f"Precision: {precision:.2f}\n"
		  f"Recall: {recall:.2f}")

	# SAVE PREPROCESSING PARAMETERS
	inference_params = {
		"preprocessing": {
			"mean": preprocessor.scaler_.mean_,
			"stdev": np.sqrt(preprocessor.scaler_.var_)
		}
	}

	with open(f"{SAVE_PATH}/inference_params.json", "w", encoding="utf-8") as f:
		json.dump(inference_params, f, ensure_ascii=False, indent=4,
				  cls=NumpyEncoder)
	return


if __name__ == "__main__":
	main()
