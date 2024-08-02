"""
This script executes cross validation to find the best model (i.e. hyperparameter
combination).

The hyperparameters to experiment include:
- The number of hidden layers
- The number of neurons in each hidden layer
- Regularisation: The dropout rate
- The activation function
- The learning rate
- The batch size

Best model from cross-validation: ANN
Model key (ID): b5e0673d-3f1b-4859-a621-95bd1d423d27
Model parameters: {'input_size': 12, 'output_size': 3, 'hidden_size': 24, 'dropout_rate': 0.15, 'n_hiddens': 2}
Optimal number of epochs: 110

Full cross validation results saved to ./cv_checkpoints1/cv_results.json.
"""

# ==============================================================================
# BEST MODEL CONFIGURATION BASED ON THIS RUN:

# RANDOM_STATE = 42
# LR = 1e-3
# BATCH_SIZE = 128
# OPTIMAL_N_EPOCHS =

# MODEL_KWARGS = {
# 	"input_size": 12
# 	"output_size": 3
# 	"hidden_size":
# 	"dropout_rate":
# 	"n_hiddens":
# }
# ==============================================================================

import random

import pandas as pd
from sklearn.preprocessing import StandardScaler

from training.cv import NeuralNetworkCrossValidator
from utils import parse
from models.ann import ANN

# ==============================================================================
# DATASET VARIABLES: CHANGE AS NEEDED
dataset_dir = 'data'  # <-- Change this as needed
dataset_name = 'fasdd_dataset'  # <-- Change this as needed
df_name = 'csv/dataset_fasdd_no_lbp_RGB.csv'  # <-- Change this as needed
split_dir = 'annotations/YOLO_CV'
train_filename = 'train.txt'
val_filename = 'val.txt'
cv_path = '../../results/cv_checkpoints_rgb'

# CHANGE MODEL HERE
MODEL = ANN  # <-- Change this as needed

# MODEL HYPERPARAMETER SETTINGS: MODEL_KWARGS need to match model kwargs exactly
RANDOM_STATE = 42
N_RANDOM_DRAWS = 50

HIDDEN_SIZES = list(range(8, 19))
N_HIDDENS = list(range(1, 4))
DROPOUT_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
ACTIVATION_FUNCTIONS = ["relu", "prelu", "silu", "selu", "tanh", "relu6", "hardswish"]
LRS = [1e-2, 1e-3, 1e-4]
BATCH_SIZES = [64, 128, 256]
N_EPOCHS = [300]

N_CV_SPLITS = 10


# GENERATE RANDOM DRAWS OF HYPERPARAMETERS
random.seed(RANDOM_STATE)
hidden_sizes = random.choices(HIDDEN_SIZES, k=N_RANDOM_DRAWS)

random.seed(RANDOM_STATE)
n_hiddens = random.choices(N_HIDDENS, k=N_RANDOM_DRAWS)

random.seed(RANDOM_STATE)
dropout_rates = random.choices(DROPOUT_RATES, k=N_RANDOM_DRAWS)

random.seed(RANDOM_STATE)
lrs = random.choices(LRS, k=N_RANDOM_DRAWS)

random.seed(RANDOM_STATE)
batch_sizes = random.choices(BATCH_SIZES, k=N_RANDOM_DRAWS)

random.seed(RANDOM_STATE)
activation_functions = random.choices(ACTIVATION_FUNCTIONS, k=N_RANDOM_DRAWS)

random.seed(RANDOM_STATE)
n_epochs = random.choices(N_EPOCHS, k=N_RANDOM_DRAWS)
# ==============================================================================

if __name__ == "__main__":
	df = pd.read_csv(f"{dataset_dir}/{dataset_name}/{df_name}")

	train_split = parse(dataset_dir, dataset_name, split_dir, train_filename)
	val_split = parse(dataset_dir, dataset_name, split_dir, val_filename)

	df_train = df[df["image"].isin(train_split)]
	df_val = df[df["image"].isin(val_split)]

	df_train = pd.concat([df_train, df_val])

	nn_cv = NeuralNetworkCrossValidator(
		n_splits=N_CV_SPLITS,
		shuffle=True,
		cv_path=cv_path
	)
	for d in range(N_RANDOM_DRAWS):
		print(f"\n Starting iteration {d+1} of {N_RANDOM_DRAWS} for randomised cross-validation search.")
		nn_cv.cross_validate(
			model=MODEL,
			model_kwargs={
				"input_size": 12,
				"output_size": 3,
				"hidden_size": hidden_sizes[d],
				"dropout_rate": dropout_rates[d],
				"n_hiddens": n_hiddens[d],
				"activation_function": activation_functions[d]
			},
			device="cpu",
			lr=lrs[d],
			batch_size=batch_sizes[d],
			n_epochs=n_epochs[d],
			# checkpoints_path='split',
			checkpoints_path=None,
			df_train=df_train,
			scaler=StandardScaler
		)
	nn_cv.get_best_model_from_cv()
