import json

from ANSCVProject.ANSCVProject.pyfiresmoke.inference import Inference
from models.ann import ANN

# ==============================================================================
# Change inference file paths here
COLOUR_SPACE = "HSV"  # <-- CHANGE THIS
ROI_OUT_PATH = "inference/inference1/inference1_rois_hsv"  # <-- CHANGE THIS
VIDEO_PATH = "inference/inference1/inference1.mp4"  # <-- CHANGE THIS
ANNOT_PATH = "inference/inference1/inference1.csv"  # <-- CHANGE THIS
CSV_OUT_DIR = "inference/inference1/inference_csv"  # <-- CHANGE THIS
# ==============================================================================


# ==============================================================================
# Change model path and args here
MODEL_SAVE_NUMBER = "1"  # <-- CHANGE THIS
MODEL_SAVE_NAME = "ANN_ep87.pt"  # <-- CHANGE THIS
MODEL_CLASS = ANN

MODEL_INFERENCE_PARAMS_PATH = f"results/final_model{MODEL_SAVE_NUMBER}/inference_params.json"
MODEL_WEIGHT_PATH = f"results/final_model{MODEL_SAVE_NUMBER}/{MODEL_SAVE_NAME}"
MODEL_CONFIG_PATH = f"results/cv_checkpoints{MODEL_SAVE_NUMBER}"
with open(f"{MODEL_CONFIG_PATH}/best_config.json", mode='r') as f:
	BEST_CONFIG = json.load(f)

MODEL_KWARGS = BEST_CONFIG["kwargs"]
# ==============================================================================


# ==============================================================================
# Inference device
INFERENCE_DEVICE = "cpu"  # <-- CHANGE THIS
# ==============================================================================


if __name__ == "__main__":
	inference_runner = Inference(
		model_class=MODEL_CLASS,
		model_kwargs=MODEL_KWARGS,
		model_weights_path=MODEL_WEIGHT_PATH,
		model_inference_params_path=MODEL_INFERENCE_PARAMS_PATH,
		data_colour_space=COLOUR_SPACE,
		roi_out_path=ROI_OUT_PATH,
		csv_out_dir=CSV_OUT_DIR,
		device=INFERENCE_DEVICE
	)
	inference_runner.infer(video_path=VIDEO_PATH, annot_path=ANNOT_PATH)
